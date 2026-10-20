"""Stage 0 driver (GPU, under run_locked.sh "A B R").
    python stage0.py smoke   # 1 train session x {short,long} x {always_A, always_B}, T1
    python stage0.py 0a      # always_B, schema T1min, v4 stage-A 10 sessions x {short,long}
    python stage0.py 0b      # train 60 sessions x {short,long} x {always_A, always_B}, T1
Resumable; run order shuffled (seed per mode); budget guard via --max-calls."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import V5, ROOT, G
import runner_v4 as RV
import httpx
from mrp.client import Server

MODES = {"smoke": 501, "0a": 502, "0b": 503}


def plan(mode):
    if mode == "0a":
        ids = json.load(open(ROOT / "followup_v4/configs/stageA_sessions.json"))["sessions"]
        src = {json.loads(l)["session_id"]: json.loads(l) for l in open(ROOT / "followup_v3/data/dev.jsonl")}
        units = [(sid, ctx, "T1min", "always_B") for sid in ids for ctx in ("short", "long")]
    else:
        src = {json.loads(l)["session_id"]: json.loads(l) for l in open(V5 / "data/train_pool.jsonl")}
        ids = json.load(open(V5 / "configs/train_harvest_sessions.json"))["sessions"]
        ids = ids[:1] if mode == "smoke" else ids
        units = [(sid, ctx, "T1", p) for sid in ids for ctx in ("short", "long") for p in ("always_A", "always_B")]
    random.Random(MODES[mode]).shuffle(units)
    return src, units


def main(mode, max_calls=10**9):
    cfg = json.load(open(ROOT / "configs/models.json"))
    http = httpx.Client()
    servers = {m: Server(m, cfg["models"][m], http) for m in ("A", "B", "R")}
    for s in servers.values():
        w = s.render_ids([{"role": "user", "content": "Write a Python function that adds two numbers."}])
        for _ in range(3): s.complete(w, max_tokens=32)
    src, units = plan(mode)
    rd = V5 / "results/stage0" / mode; rd.mkdir(parents=True, exist_ok=True)
    out = {"run_id": f"v5_{mode}", "fh": open(rd / "calls.jsonl", "a"), "sfh": open(rd / "sessions.jsonl", "a")}
    done, n = set(), 0
    for l in open(rd / "sessions.jsonl"):
        j = json.loads(l); done.add((j["session_id"], j["context"], j["schema"], j["policy"]))
    for l in open(rd / "calls.jsonl"):
        n += 1
    for i, (sid, ctx, sch, pol) in enumerate(units):
        if (sid, ctx, sch, pol) in done: continue
        if n >= max_calls - 32:
            print("BUDGET STOP", i, n, flush=True); break
        s, calls = RV.run_episode(src[sid], pol, ctx, sch, servers, out, tag={"plan_index": i, "mode": mode})
        n += len(calls)
        print(f"[{i+1}/{len(units)}] {sid} {ctx} {sch} {pol} q={s['q']:.2f} S={s['S']} calls={s['n_calls']} lat={s['latency_s']:.1f}s | calls={n}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10**9)
