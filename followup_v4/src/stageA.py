"""Stage A driver: 10 sessions x {short,long} x {T0,T1} x {always_A, always_B, follow_R0, follow_R1} = 160 runs.
Order: all 160 run units shuffled with seed 401. Budget guard: before starting a run unit, stop if A/B calls
>= 8000 - 40 or R calls >= 4000 - 40 (unit worst case 32 calls). Resumable (skips logged units)."""
import json, random, sys
from pathlib import Path
import httpx
V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4 / "src"))
import runner_v4 as RV
from mrp.client import Server

def main():
    cfg = json.load(open(RV.ROOT / "configs/models.json"))
    http = httpx.Client()
    servers = {m: Server(m, cfg["models"][m], http) for m in ("A", "B", "R")}
    for s in servers.values():
        w = s.render_ids([{"role": "user", "content": "Write a Python function that adds two numbers."}])
        for _ in range(3): s.complete(w, max_tokens=32)
    ids = json.load(open(V4 / "configs/stageA_sessions.json"))["sessions"]
    sess = {json.loads(l)["session_id"]: json.loads(l) for l in open(RV.ROOT / "followup_v3/data/dev.jsonl")}
    plan = [(sid, ctx, sch, pol) for sid in ids for ctx in ("short", "long") for sch in ("T0", "T1")
            for pol in ("always_A", "always_B", "follow_R0", "follow_R1")]
    random.Random(401).shuffle(plan)
    rd = V4 / "results/stageA"; rd.mkdir(parents=True, exist_ok=True)
    out = {"run_id": "stageA", "fh": open(rd / "calls.jsonl", "a"), "sfh": open(rd / "sessions.jsonl", "a")}
    done, nab, nr = set(), 0, 0
    for l in open(rd / "sessions.jsonl"):
        j = json.loads(l); done.add((j["session_id"], j["context"], j["schema"], j["policy"]))
    for l in open(rd / "calls.jsonl"):
        j = json.loads(l); nab += 1; nr += j["router"] is not None
    for i, (sid, ctx, sch, pol) in enumerate(plan):
        if (sid, ctx, sch, pol) in done: continue
        if nab >= 8000 - 40 or nr >= 4000 - 40:
            print("BUDGET STOP before unit", i, nab, nr, flush=True); break
        s, calls = RV.run_episode(sess[sid], pol, ctx, sch, servers, out, tag={"plan_index": i})
        nab += len(calls); nr += sum(c["router"] is not None for c in calls)
        print(f"[{i+1}/160] {sid} {ctx} {sch} {pol} q={s['q']:.2f} S={s['S']} calls={s['n_calls']} lat={s['latency_s']:.1f}s sw={s['switches']} | AB={nab} R={nr}", flush=True)

if __name__ == "__main__":
    main()
