"""B1 controlled cost replay -> profiles/cost_profile.jsonl and profiles/fit.json.

    python -m mrp.profile --transcript-run dev_smoke --out profiles

Prompts are real conversations: a dev session's request sequence with assistant turns taken from the
dev P2 transcript of that session, and the table cut to hit the target length. For each cell
(model x length x cache state x g) and repetition:
  reset prefix cache -> prepare cache (timed separately, not part of the measurement)
  -> settle 0.3 s -> measured call with forced length g (min_tokens = max_tokens = g, ignore_eos).
Cache states: none; half (token prefix = 50% of the prompt); full (the previous turn's prompt, i.e.
exactly what a model that answered the previous turn holds). Actual hits are checked vs expected.
This is a cost measurement, not a quality evaluation. Order is shuffled; 3 reps after warm-up.
"""
import argparse
import json
import random
from pathlib import Path

import httpx
import numpy as np

from . import tasks
from .client import BLOCK, Server, lcp

ROOT = Path(__file__).resolve().parents[2]
TARGET_ROWS = {"short": 20, "mid": 110, "long": 220}  # -> ~1.5K / ~6K / ~13K prompt tokens with 3 turns
GENS = [64, 320]
STATES = ["none", "half", "full"]
REPS = 3


def conversation(sess, turns, nrows):
    rows = (sess["example_rows"] + sess["extra_rows"])[:nrows]
    table = tasks.data_block(sess["domain"], rows)
    msgs = [{"role": "system", "content": tasks.SYSTEM}]
    by_step = {t["step"]: t for t in turns if t["attempt"] == 0}
    for st in sess["steps"][:3]:
        msgs.append({"role": "user", "content": (table + "\n" if st["step"] == 0 else "") + st["request"]})
        msgs.append({"role": "assistant", "content": by_step[st["step"]]["assistant_text"]})
    msgs.append({"role": "user", "content": sess["steps"][3]["request"]})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript-run", default="dev_smoke")
    ap.add_argument("--session", default="dev-S1_constant-00")
    ap.add_argument("--out", default="profiles")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cfg = json.load(open(ROOT / "configs/models.json"))
    http = httpx.Client()
    srv = {m: Server(m, cfg["models"][m], http) for m in ("A", "B")}
    sess = next(json.loads(l) for l in open(ROOT / "data/dev/smoke.jsonl") if a.session in l)
    turns = [json.loads(l) for l in open(ROOT / "results" / a.transcript_run / "turns.jsonl")]
    turns = [t for t in turns if t["session_id"] == a.session and t["policy"] == "P2"]
    prompts = {}
    for L, n in TARGET_ROWS.items():
        msgs = conversation(sess, turns, n)
        full = srv["A"].render_ids(msgs)
        prev = srv["A"].render_ids(msgs[:-2])  # previous turn's prompt
        assert full == srv["B"].render_ids(msgs)
        prompts[L] = {"full": full, "prev": prev}
        print(L, len(full), "prev", len(prev), "lcp", lcp(full, prev))
    out = Path(ROOT / a.out)
    out.mkdir(exist_ok=True)
    for m in srv.values():  # warm-up
        w = prompts["short"]["full"]
        for _ in range(3):
            m.complete(w, max_tokens=16)
    cells = [(m, L, s, g, r) for m in ("A", "B") for L in TARGET_ROWS for s in STATES for g in GENS for r in range(REPS)]
    random.Random(a.seed).shuffle(cells)
    fh = open(out / "cost_profile.jsonl", "w")
    import time
    for i, (m, L, s, g, r) in enumerate(cells):
        S, P = srv[m], prompts[L]
        S.reset_cache()
        time.sleep(0.2)
        prep = None
        if s == "half":
            pre = P["full"][: len(P["full"]) // 2]
            prep = S.complete(pre, max_tokens=1)
            expect = (len(pre) // BLOCK) * BLOCK
        elif s == "full":
            prep = S.complete(P["prev"], max_tokens=1)
            expect = (lcp(P["full"], P["prev"]) // BLOCK) * BLOCK
        else:
            expect = 0
        time.sleep(0.3)
        c = S.complete(P["full"], max_tokens=g, min_tokens=g)
        rec = {"model": m, "length": L, "state": s, "g": g, "rep": r, "prompt_tokens": len(P["full"]),
               "expected_hit": expect, "hit": c["cached_tokens"], "ttft_s": c["ttft_s"], "total_s": c["total_s"],
               "completion_tokens": c["completion_tokens"], "server": c["server"], "error": c["error"],
               "prep_s": prep["total_s"] if prep else 0.0}
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(i, m, L, s, g, r, f"hit={c['cached_tokens']}/{expect} ttft={c['ttft_s']:.3f} tot={c['total_s']:.2f}",
              flush=True)
    fh.close()
    fit(out)


def fit(out):
    recs = [json.loads(l) for l in open(out / "cost_profile.jsonl")]
    res = {}
    for m in ("A", "B"):
        R = [r for r in recs if r["model"] == m and not r["error"]]
        X, y = [], []
        for r in R:
            u = r["prompt_tokens"] - r["hit"]
            X.append([1.0, u, u * (r["hit"] + u / 2)]); y.append(r["ttft_s"])
        coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
        Xd, yd = [], []
        for r in R:
            if r["g"] == max(GENS):
                Xd.append([1.0, r["prompt_tokens"] + r["g"] / 2]); yd.append((r["total_s"] - r["ttft_s"]) / (r["g"] - 1))
        dcoef, *_ = np.linalg.lstsq(np.array(Xd), np.array(yd), rcond=None)
        pred = np.array(X) @ coef
        res[m] = {"prefill": {"a": coef[0], "b": coef[1], "k": coef[2],
                              "max_abs_rel_err": float(np.max(np.abs(pred - y) / np.array(y)))},
                  "tpot": {"d0": dcoef[0], "d1": dcoef[1]}}
    res = json.loads(json.dumps(res, default=float))
    (out / "fit_latency.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
