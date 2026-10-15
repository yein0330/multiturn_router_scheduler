"""Same-prefix A/B branch diagnostic (next call only; not a counterfactual of the whole session).

    python branch.py select-switch            # list B->A points in B2, stratified pick -> points file (before running)
    python branch.py run --points <file> --out <dir>

For a point (run, session, policy, context, cache, turn):
  1. Rebuild the exact text history the policy had at that turn from the turn log; check the rendered
     prompt hash against the logged prompt_sha (else the point is marked not reconstructable).
  2. Cache state before the routing decision: for each model m, the prompts m received earlier in that
     run since its last eviction. Restored by: reset m's prefix cache -> replay those prompts
     (max_tokens=1, timed separately as prep) -> settle 0.3 s.
  3. Measure m on the same input (greedy, max_tokens 1024, real output length). The measured hit is
     compared with the expected hit; equal => "matched state", else recorded as a mismatch.
  4. Repeat 3 times per model, restoring each time; A/B reps interleaved.
  5. Score the output with the hidden evaluator of that step. C_hat for both models uses the frozen
     cost fit with g_hat (no future length).
"""
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

FU = Path(__file__).resolve().parents[1]
ROOT = FU.parent
sys.path.insert(0, str(ROOT / "src"))
from mrp import costmodel, sandbox, tasks  # noqa: E402
from mrp.client import BLOCK, CacheTracker, Server  # noqa: E402

SESS = {json.loads(l)["session_id"]: json.loads(l) for l in open(ROOT / "data/eval/eval.jsonl")}


def load_turns(run_dir):
    T = defaultdict(list)
    for l in open(run_dir / "turns.jsonl"):
        t = json.loads(l)
        T[(t["session_id"], t["policy"], t["context"], t["cache_mode"])].append(t)
    for v in T.values():
        v.sort(key=lambda t: t["turn"])
    return T


def sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def rebuild(tok_server, sess, ts, turn, context):
    """Messages for each turn <= `turn`; returns (msgs_at_turn, prompts_by_turn, ok)."""
    table = tasks.initial_user_text(sess, context)
    msgs = [{"role": "system", "content": tasks.SYSTEM}]
    prompts, ok = {}, True
    for t in ts:
        if t["turn"] > turn:
            break
        st = sess["steps"][t["step"]]
        content = table + "\n" + st["request"] if (t["step"] == 0 and t["attempt"] == 0) else t["user_text"]
        msgs.append({"role": "user", "content": content})
        ids = tok_server.render_ids(msgs)
        prompts[t["turn"]] = ids
        ok &= sha(ids) == t["prompt_sha"]
        if t["turn"] == turn:
            return msgs, prompts, ok
        msgs.append({"role": "assistant", "content": t["assistant_text"]})
    raise ValueError("turn not found")


def cache_sets(ts, turn):
    """Prompt turns each model holds before the decision at `turn` (evictions applied)."""
    held = {"A": [], "B": []}
    for t in ts:
        if t["turn"] >= turn:
            break
        if t.get("evict"):
            held[t["evict"]["model"]] = []
        held[t["chosen"]].append(t["turn"])
    return held


# ----------------------------------------------------------------------------- selection
def select_switch(b2_runs, out_file, max_points=20):
    cands = []
    for r in b2_runs:
        T = load_turns(ROOT / "results" / r)
        for (sid, pol, ctx, cm), ts in T.items():
            if pol not in ("P4", "P6", "P7", "P8"):
                continue
            for t in ts:
                if t["router"] and t["active_before"] == "B" and t["router"]["choice"] == "A":
                    cands.append({"run": r, "session_id": sid, "policy": pol, "context": ctx, "cache": cm,
                                  "turn": t["turn"], "step": t["step"], "attempt": t["attempt"],
                                  "prompt_sha": t["prompt_sha"], "chosen": t["chosen"],
                                  "gate": t["gate"], "scenario": SESS[sid]["scenario"]})
    # drop P7/P8 duplicates of an identical P4/P6 state
    base = {(c["session_id"], c["context"], c["cache"], c["turn"], c["prompt_sha"])
            for c in cands if c["policy"] in ("P4", "P6")}
    kept = [c for c in cands if c["policy"] in ("P4", "P6") or
            (c["session_id"], c["context"], c["cache"], c["turn"], c["prompt_sha"]) not in base]
    group = lambda c: "every_turn" if c["policy"] in ("P4", "P7") else "event"
    strata = defaultdict(list)
    for c in kept:
        c["stratum"] = f"{c['context']}|{c['cache']}|{group(c)}"
        c["key"] = hashlib.sha256(f"20261018|{c['session_id']}|{c['policy']}|{c['turn']}|{c['context']}|{c['cache']}"
                                  .encode()).hexdigest()
        strata[c["stratum"]].append(c)
    for v in strata.values():
        v.sort(key=lambda c: c["key"])
    order = sorted(strata)
    picked, i = [], 0
    while len(picked) < max_points and any(strata[s] for s in order):
        s = order[i % len(order)]
        if strata[s]:
            picked.append(strata[s].pop(0))
        i += 1
    dist = defaultdict(lambda: {"candidates": 0, "after_dedup": 0, "picked": 0})
    for c in cands:
        dist[f"{c['context']}|{c['cache']}|{c['policy']}"]["candidates"] += 1
    for c in kept:
        dist[f"{c['context']}|{c['cache']}|{c['policy']}"]["after_dedup"] += 1
    for c in picked:
        dist[f"{c['context']}|{c['cache']}|{c['policy']}"]["picked"] += 1
    json.dump({"rule": json.load(open(FU / "configs/followup.json"))["switch_diag"], "b2_runs": b2_runs,
               "n_candidates": len(cands), "n_after_dedup": len(kept), "n_picked": len(picked),
               "distribution": dist, "points": picked}, open(out_file, "w"), indent=1)
    print(len(cands), len(kept), len(picked))


def select_init(out_file):
    cfg = json.load(open(FU / "configs/followup.json"))["initial_choice_diag"]
    T = load_turns(ROOT / "results/evalA")
    pts = []
    for sid in cfg["sessions"]:
        t4 = T[(sid, "P4", "short", "retain")][0]
        pts.append({"run": "evalA", "session_id": sid, "policy": "P1", "context": "short", "cache": "retain",
                    "turn": 0, "step": 0, "attempt": 0, "prompt_sha": t4["prompt_sha"],
                    "router_choice_turn0": t4["router"]["choice"], "scenario": SESS[sid]["scenario"]})
    json.dump({"rule": cfg, "points": pts}, open(out_file, "w"), indent=1)


# ----------------------------------------------------------------------------- run
def run(points_file, out_dir, reps=3):
    cfg = json.load(open(ROOT / "configs/models.json"))
    http = httpx.Client()
    srv = {m: Server(m, cfg["models"][m], http) for m in ("A", "B")}
    cm = costmodel.CostModel.load(ROOT / "profiles/fit.json")
    P = json.load(open(points_file))["points"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "measurements.jsonl", "a")
    turns_cache = {}
    for m in srv.values():  # warm-up
        w = m.render_ids([{"role": "user", "content": "Write a Python function that adds two numbers."}])
        for _ in range(3):
            m.complete(w, max_tokens=32)
    rng = random.Random(11)
    for pi, p in enumerate(P):
        if p["run"] not in turns_cache:
            turns_cache[p["run"]] = load_turns(ROOT / "results" / p["run"])
        ts = turns_cache[p["run"]][(p["session_id"], p["policy"], p["context"], p["cache"])]
        sess = SESS[p["session_id"]]
        msgs, prompts, ok = rebuild(srv["A"], sess, ts, p["turn"], p["context"])
        ids = prompts[p["turn"]]
        held = cache_sets(ts, p["turn"])
        st = sess["steps"][p["step"]]
        tr = CacheTracker()
        for m in "AB":
            for tn in held[m]:
                tr.add(m, prompts[tn])
        expect = {m: tr.predicted_hit(m, ids) for m in "AB"}
        chat = {m: cm.next_call(m, len(ids), expect[m]) for m in "AB"}
        sched = [(m, r) for r in range(reps) for m in "AB"]
        rng.shuffle(sched)
        for m, r in sched:
            S = srv[m]
            S.reset_cache()
            time.sleep(0.2)
            t0 = time.perf_counter()
            for tn in held[m]:
                S.complete(prompts[tn], max_tokens=1)
            prep_s = time.perf_counter() - t0
            time.sleep(0.3)
            c = S.complete(ids, max_tokens=1024)
            code = sandbox.extract_code(c["text"])
            ev = sandbox.evaluate(code, sess["hidden_tables"], st["gold_hidden"])
            rec = {"point": pi, **{k: p[k] for k in ("run", "session_id", "policy", "context", "cache", "turn",
                                                     "step", "attempt", "scenario")},
                   "reconstruct_ok": ok, "model": m, "rep": r, "prompt_tokens": len(ids),
                   "expected_hit": expect[m], "hit": c["cached_tokens"], "matched_state": c["cached_tokens"] == expect[m],
                   "prep_s": prep_s, "n_replayed": len(held[m]), "ttft_s": c["ttft_s"], "total_s": c["total_s"],
                   "completion_tokens": c["completion_tokens"], "finish": c["finish_reason"], "error": c["error"],
                   "pass": ev["pass"], "eval_reason": ev["reason"], "text_sha": sha(c["text"]),
                   "chat_total_s": chat[m]["total_s"], "chat_prefill_s": chat[m]["prefill_s"],
                   "chat_decode_s": chat[m]["decode_s"], "router_choice": p.get("router_choice_turn0")}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(pi, p["session_id"], p["turn"], m, r, f"hit={c['cached_tokens']}/{expect[m]} tot={c['total_s']:.2f} "
                  f"ct={c['completion_tokens']} pass={ev['pass']} ok={ok}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "select-switch":
        select_switch(sys.argv[2].split(","), sys.argv[3])
    elif cmd == "select-init":
        select_init(sys.argv[2])
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3])
