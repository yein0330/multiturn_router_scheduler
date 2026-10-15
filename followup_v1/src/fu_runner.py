"""Follow-up runner: generated from src/mrp/runner.py by followup_v1/make_runner (routing hook + plan only).

Live multi-turn policy runs.

    python -m mrp.runner --sessions data/dev/smoke.jsonl --policies P1,P2,... --context short \
        --cache retain --run-id <id> [--cost-fit profiles/fit.json]

One policy run = one session under one policy and one condition. Each run starts with all three
servers' prefix caches reset, so caches only reflect that run's own history. Each policy feeds its
own generated answers and verifier messages forward; nothing from other policies or gold leaks in.
Hidden evaluation happens after the run and is not part of the session latency.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import httpx

from mrp import costmodel, router, sandbox, tasks
from mrp.client import CacheTracker, Server

ROOT = Path(__file__).resolve().parents[2]  # repo root (followup_v1/src/ -> repo)
FU = Path(__file__).resolve().parents[1]
POLICIES = {
    "P1": "always A", "P2": "always B", "P3": "R at session start, then keep",
    "P4": "R every turn", "P5": "R at turn 0 and every k-th turn", "P6": "R at turn 0 and on events",
    "P7": "R every turn + cache gate", "P8": "R on events + cache gate",
}
K_PERIOD = 2
MAX_RETRIES = 1  # per step, only where a verifier exists (S5); identical for all policies
RETRY_SUFFIX = " Please fix `solve(rows)` and reply with the complete function."


def needs_router(policy, turn, events):
    if policy in ("P1", "P2"):
        return False
    if policy == "P3":
        return turn == 0
    if policy in ("P4", "P7"):
        return True
    if policy == "P5":
        return turn % K_PERIOD == 0
    if policy in ("P6", "P8"):
        return turn == 0 or bool(events)
    raise KeyError(policy)


# ---- follow-up policies (followup_v1/configs/followup.json); old policies delegate unchanged
K_STEP = 3
REVIEW_STEPS = 2


def route_reason(policy, turn, step, attempt, events, last_router_step):
    """Return the reason string if the router is called this turn, else None (one call max)."""
    if policy == "P9":
        return "period_step" if attempt == 0 and step % K_STEP == 0 else None
    if policy == "P10":
        if turn == 0:
            return "first_turn"
        if events:
            return "event:" + "+".join(events)
        if attempt == 0 and last_router_step is not None and step - last_router_step >= REVIEW_STEPS:
            return "review_after_2_steps"
        return None
    return "legacy" if needs_router(policy, turn, events) else None


def run_policy(sess, policy, context, cache_mode, servers, cfg, cm, out):
    for s in servers.values():
        s.reset_cache()
    time.sleep(0.3)
    tracker = CacheTracker()
    table_text = tasks.initial_user_text(sess, context)
    messages = [{"role": "system", "content": tasks.SYSTEM}]
    current, turn, pending_prefix = None, 0, ""
    final_codes, turns, t_session0 = [], [], time.perf_counter()
    prev_verdict = None
    last_router_step = None
    for st in sess["steps"]:
        attempt = 0
        while True:
            if st["step"] == 0 and attempt == 0:
                content = table_text + "\n" + st["request"]
            elif attempt == 0:
                content = pending_prefix + st["request"]
            else:
                content = prev_verdict["message"] + RETRY_SUFFIX
            messages.append({"role": "user", "content": content})
            # ---- observable events (only what is in the interface)
            events = []
            if attempt == 0 and st["kind"] == "new":
                events.append("new_task")
            if prev_verdict and prev_verdict["status"] in ("FAIL", "TOOL_ERROR", "FORMAT_ERROR") \
                    and (attempt > 0 or pending_prefix):
                events.append("verifier_" + prev_verdict["status"].lower())
            rec = {"run": out["run_id"], "session_id": sess["session_id"], "policy": policy,
                   "context": context, "cache_mode": cache_mode, "turn": turn, "step": st["step"],
                   "attempt": attempt, "events": events, "active_before": current}
            # ---- routing decision
            t_dec = time.perf_counter()
            rr, gate_rec = None, {"applied": False, "gate_s": 0.0}
            if policy == "P1":
                chosen = "A"
            elif policy == "P2":
                chosen = "B"
            elif (reason := route_reason(policy, turn, st["step"], attempt, events, last_router_step)):
                rec["route_reason"] = reason
                last_router_step = st["step"]
                rr = router.route(servers["R"], messages, table_text, current, cfg["router_decoding"]["max_tokens"])
                chosen = rr["choice"]
                if policy in ("P7", "P8"):
                    ids_probe = servers["A"].render_ids(messages)  # A and B share tokenizer + template
                    chosen, gate_rec = costmodel.gate(cm, current, chosen, ids_probe, tracker)
            else:
                chosen = current
            rec["router"] = rr
            rec["gate"] = gate_rec
            rec["chosen"] = chosen
            rec["switch"] = current is not None and chosen != current
            # ---- eviction stress condition: drop the inactive model's KV at the switch
            rec["evict"] = None
            if cache_mode == "evict" and rec["switch"]:
                rec["evict"] = {"model": current, **servers[current].reset_cache()}
                tracker.evict(current)
            t_r = time.perf_counter()
            ids = servers[chosen].render_ids(messages)
            render_s = time.perf_counter() - t_r
            pred_hit = tracker.predicted_hit(chosen, ids)
            call = servers[chosen].complete(ids, max_tokens=cfg["decoding"]["max_tokens"])
            tracker.add(chosen, ids)
            rec["render_s"] = render_s
            rec["call"] = {k: call[k] for k in ("ttft_s", "total_s", "prompt_tokens", "completion_tokens",
                                                "cached_tokens", "finish_reason", "error", "server",
                                                "kv_usage_after")}
            rec["predicted_hit"] = pred_hit
            rec["prompt_sha"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
            rec["user_text"] = content if not (st["step"] == 0 and attempt == 0) else \
                "[table]\n" + st["request"]
            rec["assistant_text"] = call["text"]
            messages.append({"role": "assistant", "content": call["text"]})
            current = chosen
            code = sandbox.extract_code(call["text"])
            rec["code_found"] = code is not None
            # ---- verifier (S5 only)
            verdict = None
            if sess["verifier"]:
                verdict = sandbox.verify(code, sess["example_rows"], st["gold_example"])
            rec["verifier"] = verdict and {k: verdict[k] for k in ("status", "exec_s", "message")}
            rec["decision_s"] = (rr["router_s"] if rr else 0.0) + gate_rec["gate_s"]
            rec["evict_s"] = rec["evict"]["s"] if rec["evict"] else 0.0
            rec["verify_s"] = verdict["exec_s"] if verdict else 0.0
            rec["turn_latency_s"] = rec["decision_s"] + rec["evict_s"] + render_s + call["total_s"] + rec["verify_s"]
            rec["turn_wall_incl_instrumentation_s"] = time.perf_counter() - t_dec
            out["fh"].write(json.dumps(rec) + "\n"); out["fh"].flush()
            turns.append(rec)
            turn += 1
            prev_verdict = verdict
            if verdict and verdict["status"] != "PASS" and attempt < MAX_RETRIES:
                attempt += 1
                continue
            pending_prefix = (verdict["message"] + "\n\n") if verdict else ""
            break
        final_codes.append(code)
    wall = time.perf_counter() - t_session0
    # ---- hidden evaluation (post hoc, excluded from latency)
    step_eval = []
    for st, code in zip(sess["steps"], final_codes):
        ev = sandbox.evaluate(code, sess["hidden_tables"], st["gold_hidden"])
        step_eval.append({"step": st["step"], "difficulty": st["difficulty"], "kind": st["kind"],
                          "op": st["op"], **{k: ev.get(k) for k in ("pass", "reason", "per_table", "detail")}})
    summ = {"run": out["run_id"], "session_id": sess["session_id"], "scenario": sess["scenario"],
            "subtype": sess["subtype"], "pattern": sess["pattern"], "structure": sess["structure"],
            "policy": policy, "context": context, "cache_mode": cache_mode,
            "n_turns": len(turns), "steps": step_eval,
            "step_pass_rate": sum(e["pass"] for e in step_eval) / len(step_eval),
            "all_pass": all(e["pass"] for e in step_eval), "final_pass": step_eval[-1]["pass"],
            "latency_s": sum(t["turn_latency_s"] for t in turns),
            "latency_excl_verifier_s": sum(t["turn_latency_s"] - t["verify_s"] for t in turns),
            "wall_incl_instrumentation_s": wall,
            "router_calls": sum(t["router"] is not None for t in turns),
            "switches": sum(t["switch"] for t in turns),
            "model_calls": {m: sum(t["chosen"] == m for t in turns) for m in ("A", "B")}}
    out["sfh"].write(json.dumps(summ) + "\n"); out["sfh"].flush()
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cost-fit", default="profiles/fit.json")
    ap.add_argument("--order-seed", type=int, default=7)
    a = ap.parse_args()
    fcfg = json.load(open(FU / "configs/followup.json"))
    cfg = json.load(open(ROOT / "configs/models.json"))
    http = httpx.Client()
    servers = {r: Server(r, cfg["models"][r], http) for r in ("A", "B", "R")}
    cm = costmodel.CostModel.load(ROOT / a.cost_fit)
    sessions = [json.loads(l) for l in open(ROOT / "data/eval/eval.jsonl")]
    rerun = set(fcfg["rerun_check"]["sessions"])
    rd = FU / "results" / a.run_id
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "args.json").write_text(json.dumps(vars(a), indent=1))
    out = {"run_id": a.run_id, "fh": open(rd / "turns.jsonl", "a"), "sfh": open(rd / "sessions.jsonl", "a")}
    done = set()
    if (rd / "sessions.jsonl").exists():
        for l in open(rd / "sessions.jsonl"):
            j = json.loads(l)
            done.add((j["session_id"], j["policy"]))
    for srv in servers.values():  # warm-up, not logged
        w = srv.render_ids([{"role": "user", "content": "Write a Python function that adds two numbers."}])
        for _ in range(3):
            srv.complete(w, max_tokens=32)
    rng = random.Random(a.order_seed)
    for sess in sessions:
        order = ["P9", "P10"] + (["P4", "P6"] if sess["session_id"] in rerun else [])
        rng.shuffle(order)
        for p in order:
            if (sess["session_id"], p) in done:
                continue
            s = run_policy(sess, p, "short", "retain", servers, cfg, cm, out)
            print(f"{sess['session_id']} {p} pass={s['step_pass_rate']:.2f} final={s['final_pass']} "
                  f"lat={s['latency_s']:.1f}s R={s['router_calls']} sw={s['switches']} calls={s['model_calls']}",
                  flush=True)


if __name__ == "__main__":
    main()
