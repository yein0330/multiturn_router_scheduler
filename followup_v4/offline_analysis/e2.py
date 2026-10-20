"""E2. Requests handled entirely by one model inside a mixed-policy run vs the same request in the always-that-model run.
'inherits' = some earlier call in the session (before this request) was made by the other model."""
import pandas as pd
from load import load_runs, index, HERE

S, C = load_runs()
run, req = index(S, C)
rows = []
for k, cs in req.items():
    src, sid, ctx, sch, pol, r = k
    if pol in ("always_A", "always_B"):
        continue
    models = {c["chosen"] for c in cs}
    if len(models) != 1:
        continue
    m = models.pop()
    earlier = [c for kk, v in req.items() if kk[:5] == k[:5] and kk[5] < r for c in v]
    inherits = any(c["chosen"] != m for c in earlier)
    ref = (src, sid, ctx, sch, "always_" + m)
    if ref not in run:
        continue
    rcs = req.get(ref + (r,), [])
    rows.append({"src": src, "schema": sch, "context": ctx, "policy": pol, "model": m, "inherits": inherits,
                 "d_calls": len(cs) - len(rcs),
                 "d_tool_errors": sum(1 for c in cs for t in c["tools"] if t["error"]) - sum(1 for c in rcs for t in c["tools"] if t["error"]),
                 "d_correct": float(run[k[:5]]["requests"][r]["correct"]) - float(run[ref]["requests"][r]["correct"]),
                 "d_latency_s": sum(c["call_latency_s"] for c in cs) - sum(c["call_latency_s"] for c in rcs)})
d = pd.DataFrame(rows)
d.to_csv(HERE / "e2_requests.csv", index=False)
g = d.groupby(["policy", "schema", "model", "inherits"]).agg(n=("d_calls", "size"), d_calls=("d_calls", "mean"),
    d_tool_errors=("d_tool_errors", "mean"), d_correct=("d_correct", "mean"), d_latency_s=("d_latency_s", "mean")).reset_index()
g.to_csv(HERE / "e2_summary.csv", index=False)
print(g.round(3).to_string())
