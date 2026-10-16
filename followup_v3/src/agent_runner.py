"""followup_v3 live agent sessions (single runner; run under run_locked.sh "A B R").

    python agent_runner.py --sessions <jsonl> --policies always_A,always_B,L0 --contexts short,long --run-id <id>
Decision point = before every model call. Per-run isolation: reset A, B, R prefix caches before each
(session, policy, context) run (outside the session clock). The fixed router R (src/mrp/router.py,
unchanged prompt) receives the conversation; tool messages are labeled ASSISTANT by router.py itself.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import httpx

V3 = Path(__file__).resolve().parents[1]
ROOT = V3.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(V3 / "src"))
from mrp import router  # noqa: E402
from mrp.client import CacheTracker, Server  # noqa: E402
import agent_tasks as G  # noqa: E402

MAX_TOKENS = 256


def render(srv, messages):
    text = srv.tok.apply_chat_template(messages, tools=G.TOOLS, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    return srv.tok(text, add_special_tokens=False)["input_ids"]


def run_session(sess, policy, context, servers, out, tag=None):
    for s in servers.values():
        s.reset_cache()
    time.sleep(0.3)
    env = G.Env(sess, context)
    tracker = CacheTracker()
    messages = [{"role": "system", "content": G.SYSTEM}]
    current, call_idx, notice = None, 0, ""
    req_results, calls = [], []
    t0 = time.perf_counter()
    for k in range(len(sess["requests"])):
        messages.append({"role": "user", "content": notice + G.request_message(sess, k)})
        notice, submitted, answer, n_in_req = "", False, None, 0
        while not submitted and n_in_req < G.MAX_CALLS_PER_REQUEST:
            rec = {"run": out["run_id"], "session_id": sess["session_id"], "policy": policy, "context": context,
                   "call": call_idx, "request": k, "call_in_request": n_in_req, "active_before": current, **(tag or {})}
            rr = None
            if policy == "always_A":
                chosen = "A"
            elif policy == "always_B":
                chosen = "B"
            elif policy == "L0":
                rr = router.route(servers["R"], messages, "", current, 4)
                chosen = rr["choice"]
            else:
                raise KeyError(policy)
            rec["router"] = rr and {k2: rr[k2] for k2 in ("choice", "parse_error", "raw", "input_tokens", "router_s")}
            rec["chosen"], rec["switch"] = chosen, current is not None and chosen != current
            tr0 = time.perf_counter()
            ids = render(servers[chosen], messages)
            render_s = time.perf_counter() - tr0
            rec["pred_hit"] = {m: tracker.predicted_hit(m, ids) for m in ("A", "B")}
            c = servers[chosen].complete(ids, max_tokens=MAX_TOKENS)
            tracker.add(chosen, ids)
            text = c["text"]
            messages.append({"role": "assistant", "content": text})
            tool_calls, perr = G.parse_tool_calls(text)
            exec_s, results = 0.0, []
            te = time.perf_counter()
            for name, args in tool_calls[: G.MAX_TOOL_CALLS_PER_TURN]:
                res = env.call(name, args)
                results.append({"name": name, "error": res.get("error"), "warning": res.get("warning")})
                messages.append({"role": "tool", "content": json.dumps(res)})
                if name == "submit" and "error" not in res:
                    submitted, answer = True, args.get("answer") if isinstance(args, dict) else None
                    break
            if not tool_calls:
                messages.append({"role": "user", "content": "[Environment] No valid tool call found"
                                 + (f" ({perr[0]})" if perr else "") + ". Respond with a tool call."})
            exec_s = time.perf_counter() - te
            rec.update({"prompt_tokens": len(ids), "prompt_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
                        "cached_tokens": c["cached_tokens"], "completion_tokens": c["completion_tokens"],
                        "ttft_s": c["ttft_s"], "total_s": c["total_s"], "finish": c["finish_reason"], "error": c["error"],
                        "server": c["server"], "render_s": render_s, "tool_exec_s": exec_s,
                        "n_tool_calls": len(tool_calls), "n_tool_calls_executed": min(len(tool_calls), G.MAX_TOOL_CALLS_PER_TURN),
                        "parse_errors": perr, "tools": results, "assistant_text": text,
                        "router_s": rr["router_s"] if rr else 0.0})
            rec["call_latency_s"] = rec["router_s"] + render_s + c["total_s"] + exec_s
            out["fh"].write(json.dumps(rec) + "\n"); out["fh"].flush()
            calls.append(rec)
            current = chosen
            call_idx += 1
            n_in_req += 1
        ok = submitted and G.correct(answer, sess["expected"][k])
        req_results.append({"request": k, "kind": sess["requests"][k]["kind"], "submitted": submitted,
                            "answer": answer, "expected": sess["expected"][k], "correct": bool(ok), "calls": n_in_req})
        if not submitted:
            notice = "[Environment] The call budget for the previous request was exhausted.\n"
    summ = {"run": out["run_id"], "session_id": sess["session_id"], "scenario": sess["scenario"],
            "pattern": sess["pattern"], "exposed": sess["exposed"], "policy": policy, "context": context,
            "requests": req_results, "q": sum(r["correct"] for r in req_results) / len(req_results),
            "S": all(r["correct"] for r in req_results), "n_calls": len(calls),
            "latency_s": sum(c["call_latency_s"] for c in calls), "wall_s": time.perf_counter() - t0,
            "router_calls": sum(c["router"] is not None for c in calls), "switches": sum(c["switch"] for c in calls),
            "model_calls": {m: sum(c["chosen"] == m for c in calls) for m in "AB"}, **(tag or {})}
    out["sfh"].write(json.dumps(summ) + "\n"); out["sfh"].flush()
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--policies", default="always_A,always_B,L0")
    ap.add_argument("--contexts", default="short,long")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--order-seed", type=int, default=5)
    a = ap.parse_args()
    cfg = json.load(open(ROOT / "configs/models.json"))
    http = httpx.Client()
    servers = {m: Server(m, cfg["models"][m], http) for m in ("A", "B", "R")}
    for s in servers.values():  # warm-up, not logged
        w = s.render_ids([{"role": "user", "content": "Write a Python function that adds two numbers."}])
        for _ in range(3):
            s.complete(w, max_tokens=32)
    rd = V3 / "results/runs" / a.run_id
    rd.mkdir(parents=True, exist_ok=True)
    out = {"run_id": a.run_id, "fh": open(rd / "calls.jsonl", "a"), "sfh": open(rd / "sessions.jsonl", "a")}
    done = set()
    for l in open(rd / "sessions.jsonl"):
        j = json.loads(l)
        done.add((j["session_id"], j["policy"], j["context"]))
    sessions = [json.loads(l) for l in open(a.sessions)]
    rng = random.Random(a.order_seed)
    for sess in sessions:
        combos = [(p, c) for p in a.policies.split(",") for c in a.contexts.split(",")]
        rng.shuffle(combos)
        for p, c in combos:
            if (sess["session_id"], p, c) in done:
                continue
            s = run_session(sess, p, c, servers, out)
            print(f"{sess['session_id']} {p} {c} q={s['q']:.2f} S={s['S']} calls={s['n_calls']} "
                  f"lat={s['latency_s']:.1f}s sw={s['switches']}", flush=True)


if __name__ == "__main__":
    main()
