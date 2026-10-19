"""followup_v4 live runner (single GPU owner; run under followup_v3/run_locked.sh "A B R").

Episode loop = followup_v3 agent_runner.run_session, parameterized by tool schema (T0/T1) and router interface
(R0/R1), plus a branch mode that starts from a restored state with a forced first model, then follows pi0.
Decision point = before every model call. Isolation resets / state restoration are outside the episode clock.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
ROOT = V4.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "followup_v3/src")); sys.path.insert(0, str(V4 / "src"))
from mrp.client import CacheTracker  # noqa: E402
import agent_tasks as G  # noqa: E402
import iface  # noqa: E402

MAX_TOKENS = 256
POLICIES = {"always_A", "always_B", "follow_R0", "follow_R1"}


def render(srv, messages, schema):
    text = srv.tok.apply_chat_template(messages, tools=iface.TOOLSETS[schema], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    return srv.tok(text, add_special_tokens=False)["input_ids"]


def sha(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]


def new_state(sess, context):
    return {"env": G.Env(sess, context), "tracker": CacheTracker(),
            "messages": [{"role": "system", "content": G.SYSTEM}], "current": None, "call_idx": 0,
            "k": 0, "n_in_req": 0, "notice": "", "req_results": [], "request_open": False}


def run_episode(sess, policy, context, schema, servers, out, state=None, first_model=None, tag=None, reset=True):
    """policy in POLICIES. Branch mode: pass a restored `state` (caches already restored by the caller) and
    `first_model` (forced model for the first call; no router call is made for it - the branch-point
    recommendation is taken from the source log and its cost is common to both branches)."""
    if state is None:
        if reset:
            for s in servers.values():
                s.reset_cache()
            time.sleep(0.3)
        state = new_state(sess, context)
    st = state
    env, tracker, messages = st["env"], st["tracker"], st["messages"]
    calls = []
    t0 = time.perf_counter()
    forced = first_model
    while st["k"] < len(sess["requests"]):
        k = st["k"]
        if not st["request_open"]:
            messages.append({"role": "user", "content": st["notice"] + G.request_message(sess, k)})
            st["notice"], st["n_in_req"], st["request_open"] = "", 0, True
            st["submitted"], st["answer"] = False, None
        while not st["submitted"] and st["n_in_req"] < G.MAX_CALLS_PER_REQUEST:
            rec = {"run": out["run_id"], "session_id": sess["session_id"], "policy": policy, "context": context,
                   "schema": schema, "call": st["call_idx"], "request": k, "call_in_request": st["n_in_req"],
                   "active_before": st["current"], **(tag or {})}
            rr = None
            if forced is not None:
                chosen, rec["forced_first"] = forced, True
                forced = None
            elif policy == "always_A":
                chosen = "A"
            elif policy == "always_B":
                chosen = "B"
            else:
                rr = iface.route(policy.split("_")[1], servers["R"], messages, st["current"])
                chosen = rr["choice"]
            rec["router"] = rr and {k2: rr[k2] for k2 in ("choice", "parse_error", "raw", "input_tokens", "router_s")}
            rec["chosen"], rec["switch"] = chosen, st["current"] is not None and chosen != st["current"]
            tr0 = time.perf_counter()
            ids = render(servers[chosen], messages, schema)
            render_s = time.perf_counter() - tr0
            rec["pred_hit"] = {m: tracker.predicted_hit(m, ids) for m in ("A", "B")}
            c = servers[chosen].complete(ids, max_tokens=MAX_TOKENS)
            tracker.add(chosen, ids)
            text = c["text"]
            messages.append({"role": "assistant", "content": text})
            tool_calls, perr = G.parse_tool_calls(text)
            results = []
            te = time.perf_counter()
            for name, args in tool_calls[: G.MAX_TOOL_CALLS_PER_TURN]:
                res = env.call(name, args)
                results.append({"name": name, "error": res.get("error"), "warning": res.get("warning"), "args": args})
                messages.append({"role": "tool", "content": json.dumps(res)})
                if name == "submit" and "error" not in res:
                    st["submitted"], st["answer"] = True, args.get("answer") if isinstance(args, dict) else None
                    break
            if not tool_calls:
                messages.append({"role": "user", "content": "[Environment] No valid tool call found"
                                 + (f" ({perr[0]})" if perr else "") + ". Respond with a tool call."})
            exec_s = time.perf_counter() - te
            rec.update({"prompt_tokens": len(ids), "prompt_sha": sha(ids), "cached_tokens": c["cached_tokens"],
                        "completion_tokens": c["completion_tokens"], "ttft_s": c["ttft_s"], "total_s": c["total_s"],
                        "finish": c["finish_reason"], "error": c["error"], "render_s": render_s, "tool_exec_s": exec_s,
                        "n_tool_calls": len(tool_calls), "parse_errors": perr, "tools": results, "assistant_text": text,
                        "router_s": rr["router_s"] if rr else 0.0})
            rec["call_latency_s"] = rec["router_s"] + render_s + c["total_s"] + exec_s
            out["fh"].write(json.dumps(rec) + "\n"); out["fh"].flush()
            calls.append(rec)
            st["current"] = chosen
            st["call_idx"] += 1
            st["n_in_req"] += 1
        ok = st["submitted"] and G.correct(st["answer"], sess["expected"][k])
        st["req_results"].append({"request": k, "kind": sess["requests"][k]["kind"], "submitted": st["submitted"],
                                  "answer": st["answer"], "expected": sess["expected"][k], "correct": bool(ok),
                                  "calls": st["n_in_req"]})
        if not st["submitted"]:
            st["notice"] = "[Environment] The call budget for the previous request was exhausted.\n"
        st["request_open"] = False
        st["k"] += 1
    rr_ = st["req_results"]
    summ = {"run": out["run_id"], "session_id": sess["session_id"], "scenario": sess["scenario"], "pattern": sess["pattern"],
            "exposed": sess["exposed"], "policy": policy, "context": context, "schema": schema, "requests": rr_,
            "q": sum(r["correct"] for r in rr_) / len(rr_), "S": all(r["correct"] for r in rr_),
            "n_calls": len(calls), "latency_s": sum(c["call_latency_s"] for c in calls), "wall_s": time.perf_counter() - t0,
            "router_calls": sum(c["router"] is not None for c in calls), "switches": sum(c["switch"] for c in calls),
            "model_calls": {m: sum(c["chosen"] == m for c in calls) for m in "AB"},
            "router_s": sum(c["router_s"] for c in calls), **(tag or {})}
    out["sfh"].write(json.dumps(summ) + "\n"); out["sfh"].flush()
    return summ, calls
