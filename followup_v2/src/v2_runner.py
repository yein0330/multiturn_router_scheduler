"""followup_v2 runner: copy of src/mrp/runner.py with (1) fixed switch-step policies and
(2) optional start from a restored branch state. Everything else is unchanged.

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

ROOT = Path(__file__).resolve().parents[2]
POLICIES = {
    "P1": "always A", "P2": "always B", "P3": "R at session start, then keep",
    "P4": "R every turn", "P5": "R at turn 0 and every k-th turn", "P6": "R at turn 0 and on events",
    "P7": "R every turn + cache gate", "P8": "R on events + cache gate",
}
K_PERIOD = 2
MAX_RETRIES = 1  # per step, only where a verifier exists (S5); identical for all policies
RETRY_SUFFIX = " Please fix `solve(rows)` and reply with the complete function."
# fixed switch policies (followup_v2/configs/v2.json): B from task step s onward, all attempts
FIXED = {"BBBBB": 0, "ABBBB": 1, "AABBB": 2, "AAABB": 3, "AAAAB": 4, "AAAAA": 5, "stayA": 99, "switchB": 0}


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


def run_policy(sess, policy, context, cache_mode, servers, cfg, cm, out, start=None, tag=None):
    """start (branch mode): dict(messages, tracker, current, turn, step, pending_prefix, prev_verdict);
    the caller has already restored the server caches (restore cost is measured by the caller)."""
    table_text = tasks.initial_user_text(sess, context)
    if start is None:
        for s in servers.values():
            s.reset_cache()
        time.sleep(0.3)
        tracker = CacheTracker()
        messages = [{"role": "system", "content": tasks.SYSTEM}]
        current, turn, pending_prefix, prev_verdict, first_step = None, 0, "", None, 0
    else:
        tracker, messages = start["tracker"], list(start["messages"])
        current, turn, pending_prefix = start["current"], start["turn"], start["pending_prefix"]
        prev_verdict, first_step = start["prev_verdict"], start["step"]
    final_codes, turns, t_session0 = [], [], time.perf_counter()
    for st in sess["steps"][first_step:]:
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
                   "attempt": attempt, "events": events, "active_before": current, **(tag or {})}
            # ---- routing decision
            t_dec = time.perf_counter()
            rr, gate_rec = None, {"applied": False, "gate_s": 0.0}
            if policy in FIXED:
                chosen = "B" if st["step"] >= FIXED[policy] else "A"
            elif policy == "P1":
                chosen = "A"
            elif policy == "P2":
                chosen = "B"
            elif needs_router(policy, turn, events):
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
    for st, code in zip(sess["steps"][first_step:], final_codes):
        ev = sandbox.evaluate(code, sess["hidden_tables"], st["gold_hidden"])
        step_eval.append({"step": st["step"], "difficulty": st["difficulty"], "kind": st["kind"],
                          "op": st["op"], **{k: ev.get(k) for k in ("pass", "reason", "per_table", "detail")}})
    summ = {**(tag or {}), "run": out["run_id"], "session_id": sess["session_id"], "scenario": sess["scenario"],
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


