"""followup_v2 driver (single GPU owner; run under flock via run_v2.sh).

    python v2_main.py baseline      # BBBBB/AAAAA on 10 seed-picked sessions x 2 conditions (vs P2/P1)
    python v2_main.py decide        # apply the frozen reuse rule -> results/baseline_decision.json
    python v2_main.py main          # 4 switch policies (+ BBBBB/AAAAA if reuse rejected) x 50 x 2 conditions
    python v2_main.py branch        # same-history branch: 10 sessions x steps{1,3} x 2 conditions x 2 paths x 3 reps
Runs are resumable: (session, policy, context[, point, path, rep]) already logged are skipped.
"""
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V2 / "src"))
from mrp import sandbox, tasks  # noqa: E402
from mrp.client import CacheTracker, Server  # noqa: E402
import v2_runner as R  # noqa: E402

CFG = json.load(open(V2 / "configs/v2.json"))
SESS = [json.loads(l) for l in open(ROOT / "data/eval/eval.jsonl")]
BY_ID = {s["session_id"]: s for s in SESS}
OLD = {"short": ROOT / "results/evalA", "long": ROOT / "results/evalB_long_retain"}


def servers():
    cfg = json.load(open(ROOT / "configs/models.json"))
    http = httpx.Client()
    srv = {m: Server(m, cfg["models"][m], http) for m in ("A", "B")}
    for s in srv.values():  # warm-up, not logged
        w = s.render_ids([{"role": "user", "content": "Write a Python function that adds two numbers."}])
        for _ in range(3):
            s.complete(w, max_tokens=32)
    return cfg, srv


def outputs(name):
    d = V2 / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    done = set()
    if (d / "sessions.jsonl").exists():
        for l in open(d / "sessions.jsonl"):
            j = json.loads(l)
            done.add(tuple(j.get(k) for k in ("session_id", "policy", "context", "point", "path", "rep")))
    return {"run_id": name, "fh": open(d / "turns.jsonl", "a"), "sfh": open(d / "sessions.jsonl", "a")}, done


def run_plan(name, plan):
    cfg, srv = servers()
    out, done = outputs(name)
    for i, (sid, pol, ctx) in enumerate(plan):
        if (sid, pol, ctx, None, None, None) in done:
            continue
        s = R.run_policy(BY_ID[sid], pol, ctx, "retain", srv, cfg, None, out,
                         tag={"plan_index": i, "t_start": time.strftime("%H:%M:%S")})
        print(f"[{i+1}/{len(plan)}] {sid} {pol} {ctx} pass={s['step_pass_rate']:.2f} lat={s['latency_s']:.1f}s",
              flush=True)


def baseline():
    plan = [(sid, p, c) for sid in CFG["baseline_check"]["sessions"] for p in ("BBBBB", "AAAAA")
            for c in ("short", "long")]
    random.Random(30).shuffle(plan)
    run_plan("baseline_check", plan)


def load_old():
    S, T = {}, defaultdict(list)
    for ctx, d in OLD.items():
        for l in open(d / "sessions.jsonl"):
            j = json.loads(l)
            S[(j["session_id"], j["policy"], ctx)] = j
        for l in open(d / "turns.jsonl"):
            j = json.loads(l)
            T[(j["session_id"], j["policy"], ctx)].append(j)
    return S, T


def decide():
    S, T = load_old()
    new = [json.loads(l) for l in open(V2 / "results/baseline_check/sessions.jsonl")]
    newT = defaultdict(list)
    for l in open(V2 / "results/baseline_check/turns.jsonl"):
        j = json.loads(l)
        newT[(j["session_id"], j["policy"], j["context"])].append(j)
    m = {"BBBBB": "P2", "AAAAA": "P1"}
    rows, ratios = [], defaultdict(list)
    for s in new:
        o = S[(s["session_id"], m[s["policy"]], s["context"])]
        nt = sorted(newT[(s["session_id"], s["policy"], s["context"])], key=lambda t: t["turn"])
        ot = sorted(T[(s["session_id"], m[s["policy"]], s["context"])], key=lambda t: t["turn"])
        same_text = len(nt) == len(ot) and all(a["assistant_text"] == b["assistant_text"] for a, b in zip(nt, ot))
        same_steps = [e["pass"] for e in s["steps"]] == [e["pass"] for e in o["steps"]]
        r = s["latency_s"] / o["latency_s"]
        ratios[s["context"]].append(r)
        rows.append({"session_id": s["session_id"], "policy": s["policy"], "context": s["context"],
                     "same_text": same_text, "same_step_outcomes": same_steps, "latency_new": s["latency_s"],
                     "latency_old": o["latency_s"], "ratio": r})
    n_same = sum(r["same_step_outcomes"] for r in rows)
    med = {c: float(np.median(v)) for c, v in ratios.items()}
    ok = n_same >= 38 and all(0.95 <= v <= 1.05 for v in med.values())
    res = {"rule": CFG["baseline_check"]["reuse_rule"], "n": len(rows), "same_step_outcomes": n_same,
           "same_text": sum(r["same_text"] for r in rows), "median_latency_ratio": med,
           "ratio_range": {c: [float(min(v)), float(max(v))] for c, v in ratios.items()},
           "reuse_old_baselines": ok, "rows": rows}
    json.dump(res, open(V2 / "results/baseline_decision.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


def main_run():
    dec = json.load(open(V2 / "results/baseline_decision.json"))
    pols = CFG["main_run"]["policies"] + ([] if dec["reuse_old_baselines"] else ["BBBBB", "AAAAA"])
    rng = random.Random(31)
    plan = []
    for s in SESS:
        combos = [(s["session_id"], p, c) for p in pols for c in ("short", "long")]
        rng.shuffle(combos)
        plan += combos
    run_plan("main", plan)


# ----------------------------------------------------------------------------- branch
def sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def branch_points():
    _, T = load_old()
    pts = []
    for sid in CFG["branch"]["sessions"]:
        for ctx in ("short", "long"):
            ts = sorted(T[(sid, "P1", ctx)], key=lambda t: t["turn"])
            for k in (1, 3):
                bt = next(t for t in ts if t["step"] == k and t["attempt"] == 0)
                pts.append({"session_id": sid, "context": ctx, "step": k, "turn": bt["turn"]})
    return pts, T


def restore(srv, sess, ts, pt):
    """Rebuild history before the branch turn, restore A's cache, B empty. Returns (start, info)."""
    table = tasks.initial_user_text(sess, pt["context"])
    msgs = [{"role": "system", "content": tasks.SYSTEM}]
    prompts, ok = {}, True
    for t in ts:
        st = sess["steps"][t["step"]]
        content = table + "\n" + st["request"] if (t["step"] == 0 and t["attempt"] == 0) else t["user_text"]
        msgs.append({"role": "user", "content": content})
        ids = srv["A"].render_ids(msgs)
        prompts[t["turn"]] = ids
        ok &= sha(ids) == t["prompt_sha"]
        if t["turn"] == pt["turn"]:
            msgs.pop()
            break
        msgs.append({"role": "assistant", "content": t["assistant_text"]})
    bt = next(t for t in ts if t["turn"] == pt["turn"])
    req = sess["steps"][pt["step"]]["request"]
    assert bt["user_text"].endswith(req)
    prev = next(t for t in ts if t["turn"] == pt["turn"] - 1)
    prev_verdict = None
    if prev["verifier"]:
        prev_verdict = {"status": prev["verifier"]["status"], "message": prev["verifier"]["message"]}
    for s in srv.values():
        s.reset_cache()
    time.sleep(0.2)
    t0 = time.perf_counter()
    tracker = CacheTracker()
    for t in ts:
        if t["turn"] >= pt["turn"]:
            break
        srv["A"].complete(prompts[t["turn"]], max_tokens=1)
        tracker.add("A", prompts[t["turn"]])
    prep = time.perf_counter() - t0
    time.sleep(0.3)
    start = {"messages": msgs, "tracker": tracker, "current": "A", "turn": pt["turn"], "step": pt["step"],
             "pending_prefix": bt["user_text"][: len(bt["user_text"]) - len(req)], "prev_verdict": prev_verdict}
    exp = {m: tracker.predicted_hit(m, prompts[pt["turn"]]) for m in "AB"}
    return start, {"reconstruct_ok": ok, "prep_s": prep, "expected_first_hit": exp,
                   "branch_prompt_sha": sha(prompts[pt["turn"]])}


def branch():
    pts, T = branch_points()
    json.dump(pts, open(V2 / "configs/branch_points.json", "w"), indent=1)
    sched = [(i, path, rep) for i in range(len(pts)) for path in ("stayA", "switchB") for rep in range(3)]
    random.Random(41).shuffle(sched)
    cfg, srv = servers()
    out, done = outputs("branch")
    for j, (i, path, rep) in enumerate(sched):
        pt = pts[i]
        if (pt["session_id"], path, pt["context"], i, path, rep) in done:
            continue
        sess = BY_ID[pt["session_id"]]
        ts = sorted(T[(pt["session_id"], "P1", pt["context"])], key=lambda t: t["turn"])
        start, info = restore(srv, sess, ts, pt)
        tag = {"point": i, "path": path, "rep": rep, "branch_step": pt["step"], "branch_turn": pt["turn"], **info}
        s = R.run_policy(sess, path, pt["context"], "retain", srv, cfg, None, out, start=start, tag=tag)
        print(f"[{j+1}/{len(sched)}] pt{i} {pt['session_id']} {pt['context']} step{pt['step']} {path} r{rep} "
              f"ok={info['reconstruct_ok']} pass={s['step_pass_rate']:.2f} final={s['final_pass']} "
              f"lat={s['latency_s']:.1f}s", flush=True)


if __name__ == "__main__":
    {"baseline": baseline, "decide": decide, "main": main_run, "branch": branch}[sys.argv[1]]()
