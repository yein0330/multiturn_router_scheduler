"""E4. Predicting remaining work within the current request from cheap features (session-level CV, linear model)."""
import collections, json
import numpy as np, pandas as pd
from load import load_runs, index, tool_sig, r2, HERE

S, C = load_runs()
run, req = index(S, C)
rows = []
for k, cs in req.items():
    src, sid, ctx, sch, pol, r = k
    prev_sig, rep = None, 0
    for i, c in enumerate(cs):
        sig = tool_sig(c)
        rows.append({"data": f"{src}_{sch}", "session_id": sid, "context": ctx, "pos": i, "model_B": c["chosen"] == "B",
                     "prompt_k": c["prompt_tokens"] / 1000, "hit_k": (c["cached_tokens"] or 0) / 1000,
                     "prev_err": i > 0 and any(t["error"] for t in cs[i - 1]["tools"]),
                     "rep_so_far": rep, "prev_out": cs[i - 1]["completion_tokens"] if i else 0, "request": r,
                     "long": ctx == "long", "y_calls": len(cs) - i, "y_lat": sum(x["call_latency_s"] for x in cs[i:])})
        if sig and prev_sig is not None and sig[0] == prev_sig:
            rep += 1
        prev_sig = sig[-1] if sig else prev_sig
d = pd.DataFrame(rows)
BASE = ["pos", "model_B"]
CHEAP = BASE + ["prompt_k", "hit_k", "prev_err", "rep_so_far", "prev_out", "request", "long"]


def cv(g, feats, y, k=5):
    sids = sorted(g.session_id.unique())
    rng = np.random.default_rng(0); rng.shuffle(sids)
    folds = {s: i % k for i, s in enumerate(sids)}
    pred = np.zeros(len(g))
    X = np.column_stack([np.ones(len(g))] + [g[f].astype(float).values for f in feats])
    fold = g.session_id.map(folds).values
    for f in range(k):
        tr, te = fold != f, fold == f
        w, *_ = np.linalg.lstsq(X[tr], g[y].values[tr], rcond=None)
        pred[te] = X[te] @ w
    return r2(g[y].values, pred)


out = []
for data, g in d.groupby("data"):
    if data == "v4A_T0":
        continue
    g = g.reset_index(drop=True)
    for y in ("y_calls", "y_lat"):
        out.append({"data": data, "target": y, "n": len(g), "R2_cheap": cv(g, CHEAP, y), "R2_base": cv(g, BASE, y)})
o = pd.DataFrame(out); o.to_csv(HERE / "e4_r2.csv", index=False)
print(o.round(3).to_string())
