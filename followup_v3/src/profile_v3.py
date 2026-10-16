"""Stage 1-3 cost profile grid (pure-system: identical token prefix, only cache state changes).

    python profile_v3.py measure --transcripts <calls.jsonl> --out <dir>   (needs A, B servers; under flock)
    python profile_v3.py fit --out <dir>                                   (CPU)
Grid: context {1K,4K,8K,16K,24K} x reuse {0,0.5,0.95} x output {16,48,128,320} x model {A,B} x 3 reps.
Prompt: real agent-session messages (system + several smoke sessions' user/assistant/tool messages,
concatenated in order) cut at the last message boundary <= target, then the generation prompt.
Cache state: reset -> prefill token prefix floor(r*L) (block-floored by the engine) -> settle -> measure
with forced output length (min_tokens = max_tokens = g, ignore_eos). This setting is for cost calibration
only, never for task evaluation. Output cap 320 here vs 256 in tasks (documented).
Hold-out: all cells with context 8K or output 48 (cell-level split; C3 never reads them).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np

V3 = Path(__file__).resolve().parents[1]
ROOT = V3.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(V3 / "src"))
from mrp.client import Server  # noqa: E402
import agent_tasks as G  # noqa: E402

CTX = [1024, 4096, 8192, 16384, 24576]
REUSE = [0.0, 0.5, 0.95]
OUTS = [16, 48, 128, 320]
REPS = 3
HOLDOUT = lambda L, g: L == 8192 or g == 48


def build_prompts(srv, transcripts):
    """Concatenate messages from several logged sessions (real content) and cut to targets."""
    by = {}
    for l in open(transcripts):
        c = json.loads(l)
        if c["context"] == "long":
            by.setdefault((c["session_id"], c["policy"]), []).append(c)
    msgs = [{"role": "system", "content": G.SYSTEM}]
    sess = {json.loads(l)["session_id"]: json.loads(l) for l in open(V3 / "data/smoke.jsonl")}
    for (sid, pol), cs in sorted(by.items()):
        if pol != "always_B":
            continue
        s = sess[sid]
        env = G.Env(s, "long")
        for k in range(len(s["requests"])):
            msgs.append({"role": "user", "content": G.request_message(s, k)})
            for c in [c for c in sorted(cs, key=lambda c: c["call"]) if c["request"] == k]:
                msgs.append({"role": "assistant", "content": c["assistant_text"]})
                calls, _ = G.parse_tool_calls(c["assistant_text"])
                for name, args in calls[: G.MAX_TOOL_CALLS_PER_TURN]:
                    msgs.append({"role": "tool", "content": json.dumps(env.call(name, args))})
    # candidate cut points: after each user/tool message
    out = {}
    lens = []
    for i in range(2, len(msgs) + 1):
        if msgs[i - 1]["role"] in ("user", "tool"):
            text = srv.tok.apply_chat_template(msgs[:i], tools=G.TOOLS, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
            lens.append((i, len(srv.tok(text, add_special_tokens=False)["input_ids"])))
    for L in CTX:
        i, n = max([x for x in lens if x[1] <= L], key=lambda x: x[1])
        text = srv.tok.apply_chat_template(msgs[:i], tools=G.TOOLS, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        out[L] = srv.tok(text, add_special_tokens=False)["input_ids"]
    return out, lens[-1][1]


def measure(a):
    cfg = json.load(open(ROOT / "configs/models.json"))
    http = httpx.Client()
    srv = {m: Server(m, cfg["models"][m], http) for m in ("A", "B")}
    prompts, maxlen = build_prompts(srv["A"], a.transcripts)
    assert all(srv["A"].tok.decode(p) == srv["B"].tok.decode(p) for p in prompts.values())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"targets": CTX, "actual_len": {L: len(p) for L, p in prompts.items()}, "max_available": maxlen,
               "holdout_rule": "context==8192 or output==48", "forced_output": "min_tokens=max_tokens=g, ignore_eos=True"},
              open(out / "profile_split_manifest.json", "w"), indent=1)
    for m in srv.values():
        for _ in range(3):
            m.complete(prompts[1024], max_tokens=16)
    cells = [(m, L, r, g, k) for m in "AB" for L in CTX for r in REUSE for g in OUTS for k in range(REPS)]
    random.Random(17).shuffle(cells)
    fh = open(out / "profile_calls.jsonl", "a")
    for i, (m, L, r, g, k) in enumerate(cells):
        S, P = srv[m], prompts[L]
        S.reset_cache()
        time.sleep(0.2)
        prep = None
        npre = int(r * len(P))
        if npre > 0:
            prep = S.complete(P[:npre], max_tokens=1)
        time.sleep(0.3)
        c = S.complete(P, max_tokens=g, min_tokens=g)
        exp = (min(npre // 16, (len(P) - 1) // 16) * 16) if npre else 0
        rec = {"model": m, "ctx_target": L, "ctx": len(P), "reuse": r, "g": g, "rep": k, "expected_hit": exp,
               "hit": c["cached_tokens"], "ttft_s": c["ttft_s"], "total_s": c["total_s"],
               "completion_tokens": c["completion_tokens"], "prep_s": prep["total_s"] if prep else 0.0,
               "error": c["error"], "holdout": HOLDOUT(L, g)}
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        if i % 30 == 0:
            print(i, m, L, r, g, f"hit={c['cached_tokens']}/{exp} tot={c['total_s']:.2f}", flush=True)


# ----------------------------------------------------------------------------- cost representations
def feats(kind, ell, c, y):
    n = ell - c
    if kind == "C0":
        return np.array([1.0, ell, y])
    if kind == "C1":
        return np.array([1.0, n, y])
    if kind == "C2":
        return np.array([1.0, n, n * (c + n / 2), y, y * (ell + y / 2)])
    raise KeyError(kind)


class CostModels:
    def __init__(self, fit, grid):
        self.fit, self.grid = fit, grid

    def predict(self, kind, m, ell, c, y):
        if kind == "C3":
            return self.c3(m, ell, c, y)
        return float(np.dot(self.fit[m][kind], feats(kind, ell, c, y)))

    def c3(self, m, ell, c, y):
        """Trilinear interpolation over (ctx, reuse fraction, output) on fit cells; outside range -> C2."""
        g = self.grid[m]
        xs, rs, ys = g["ctx"], g["reuse"], g["out"]
        r = c / ell if ell else 0.0
        if not (xs[0] <= ell <= xs[-1] and ys[0] <= y <= ys[-1]):
            return self.predict("C2", m, ell, c, y)
        r = min(max(r, rs[0]), rs[-1])
        V = np.array(g["val"])  # [ctx][reuse][out]

        def bracket(arr, v):
            j = int(np.searchsorted(arr, v, side="right") - 1)
            j = min(max(j, 0), len(arr) - 2)
            t = (v - arr[j]) / (arr[j + 1] - arr[j])
            return j, t
        i, ti = bracket(xs, ell); j, tj = bracket(rs, r); k, tk = bracket(ys, y)
        acc = 0.0
        for di, wi in ((0, 1 - ti), (1, ti)):
            for dj, wj in ((0, 1 - tj), (1, tj)):
                for dk, wk in ((0, 1 - tk), (1, tk)):
                    acc += wi * wj * wk * V[i + di][j + dj][k + dk]
        return float(acc)

    @classmethod
    def load(cls, path):
        j = json.load(open(path))
        return cls(j["fit"], j["grid"])


def fit_models(a):
    out = Path(a.out)
    R = [json.loads(l) for l in open(out / "profile_calls.jsonl")]
    R = [r for r in R if not r["error"]]
    fit, grid, rep = {}, {}, {}
    for m in "AB":
        tr = [r for r in R if r["model"] == m and not r["holdout"]]
        te = [r for r in R if r["model"] == m and r["holdout"]]
        fit[m] = {}
        for kind in ("C0", "C1", "C2"):
            X = np.array([feats(kind, r["ctx"], r["hit"], r["completion_tokens"]) for r in tr])
            y = np.array([r["total_s"] for r in tr])
            w, *_ = np.linalg.lstsq(X, y, rcond=None)
            fit[m][kind] = w.tolist()
        ctxs = sorted({r["ctx"] for r in tr}); outs = sorted({r["g"] for r in tr})
        val = [[[float(np.median([r["total_s"] for r in tr if r["ctx"] == L and r["reuse"] == rr and r["g"] == g]))
                 for g in outs] for rr in REUSE] for L in ctxs]
        grid[m] = {"ctx": ctxs, "reuse": REUSE, "out": outs, "val": val}
    cm = CostModels(fit, grid)
    for m in "AB":
        te = [r for r in R if r["model"] == m and r["holdout"]]
        tr = [r for r in R if r["model"] == m and not r["holdout"]]
        for kind in ("C0", "C1", "C2", "C3"):
            for nm, S in (("fit", tr), ("holdout", te)):
                e = np.array([cm.predict(kind, m, r["ctx"], r["hit"], r["completion_tokens"]) - r["total_s"] for r in S])
                rep.setdefault(m, {}).setdefault(kind, {})[nm] = {"mae_s": float(np.abs(e).mean()), "p95_abs_s": float(np.percentile(np.abs(e), 95)),
                                                                   "n": len(S)}
        rep[m]["hit_eq_expected"] = sum(r["hit"] == r["expected_hit"] for r in R if r["model"] == m)
        rep[m]["n_calls"] = sum(r["model"] == m for r in R)
        rep[m]["coef_C2"] = dict(zip(["eta", "alpha", "beta", "gamma", "delta"], fit[m]["C2"]))
        rep[m]["negative_coef_flags"] = {k: [i for i, v in enumerate(fit[m][k]) if v < 0] for k in ("C0", "C1", "C2")}
    json.dump({"fit": fit, "grid": grid}, open(out / "cost_models.json", "w"), indent=1)
    json.dump(rep, open(out / "cost_model_errors.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--transcripts", default=str(V3 / "results/runs/smoke/calls.jsonl"))
    ap.add_argument("--out", default=str(V3 / "results/stage01/profile"))
    a = ap.parse_args()
    {"measure": measure, "fit": fit_models}[a.cmd](a)
