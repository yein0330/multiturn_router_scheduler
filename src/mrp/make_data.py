"""Write dev / eval session files + manifest. Deterministic from seeds.

    python -m mrp.make_data dev
    python -m mrp.make_data eval      # only at the freeze step
Seed ranges: dev smoke 1000+, dev calibration 2000+, dev single-step 3000+, eval 50000+.
Dev uses domains {orders, library} and phrasing variant 0; eval uses {clinic, fleet, warehouse} and variant 1.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

from . import tasks

ROOT = Path(__file__).resolve().parents[2]
SC_OFFSET = {s: i * 100 for i, s in enumerate(tasks.SCENARIOS)}


def single(split, diff, idx, seed):
    """One-step session (a new task) used only for the dev difficulty check."""
    rng = random.Random(seed)
    doms = [k for k, v in tasks.DOMAINS.items() if v["split"] == split]
    dom = doms[idx % len(doms)]
    example = tasks.gen_rows(dom, 12, rng)
    hidden = [tasks.gen_rows(dom, 40, rng), tasks.gen_rows(dom, 60, rng), tasks.gen_rows(dom, 30, rng, ties=True)]
    spec, text, meta = tasks.make_step(dom, "new", diff, None, rng, 0 if split == "dev" else 1)
    st = {"step": 0, "request": text, "spec": spec, **meta,
          "gold_hidden": [tasks.spec_gold(dom, spec, t) for t in hidden],
          "gold_example": tasks.spec_gold(dom, spec, example)}
    return {"session_id": f"{split}-single-{diff}-{idx:02d}", "split": split, "scenario": "single",
            "index": idx, "seed": seed, "domain": dom, "phrasing_variant": 0, "subtype": diff,
            "pattern": "E" if diff == "easy" else "H", "structure": "N", "verifier": False,
            "example_rows": example, "extra_rows": [], "hidden_tables": hidden, "steps": [st]}


def write(path, sessions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")
    return {"file": str(path.relative_to(ROOT)), "n": len(sessions),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ids_seeds": [[s["session_id"], s["seed"]] for s in sessions]}


def main(which):
    man = {}
    if which == "dev":
        smoke = [tasks.build_session("dev", sc, i, 1000 + SC_OFFSET[sc] + i) for sc in tasks.SCENARIOS for i in range(2)]
        calib = [tasks.build_session("dev", sc, i, 2000 + SC_OFFSET[sc] + i) for sc in tasks.SCENARIOS for i in range(4)]
        for s in calib:
            s["session_id"] = s["session_id"].replace("dev-", "devcal-")
        sing = [single("dev", d, i, 3000 + (0 if d == "easy" else 500) + i) for d in ("easy", "hard") for i in range(24)]
        man["smoke"] = write(ROOT / "data/dev/smoke.jsonl", smoke)
        man["calib_sessions"] = write(ROOT / "data/dev/calib_sessions.jsonl", calib)
        man["calib_single"] = write(ROOT / "data/dev/calib_single.jsonl", sing)
    else:
        ev = [tasks.build_session("eval", sc, i, 50000 + SC_OFFSET[sc] + i) for sc in tasks.SCENARIOS for i in range(10)]
        man["eval"] = write(ROOT / "data/eval/eval.jsonl", ev)
    (ROOT / f"data/{which}/manifest.json").write_text(json.dumps(man, indent=1))
    for k, v in man.items():
        print(k, v["n"], v["sha256"][:12])


if __name__ == "__main__":
    main(sys.argv[1])
