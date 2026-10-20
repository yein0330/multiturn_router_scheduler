"""E1. Cache cost of re-processing history at switch points (C2 prefill model; estimate, not measured)."""
import collections, json
import numpy as np, pandas as pd
from load import load_runs, prefill_c2, HERE

S, C = load_runs()
rows = []
for c in C:
    if c["active_before"] is None or c["policy"] in ("always_A", "always_B"):
        continue
    m, ell, hit = c["chosen"], c["prompt_tokens"], c["cached_tokens"] or 0
    lost = prefill_c2(m, ell, 0) - prefill_c2(m, ell - hit, hit)   # extra prefill if the target's cache were lost
    rows.append({"src": c["src"], "schema": c["schema"], "context": c["context"], "policy": c["policy"],
                 "kind": "switch" if c["switch"] else "stay", "to": m, "hit_frac": hit / ell,
                 "lost_prefill_s": lost, "call_s": c["call_latency_s"]})
d = pd.DataFrame(rows)
d.to_csv(HERE / "e1_calls.csv", index=False)
g = d.groupby(["src", "schema", "context", "kind"]).agg(n=("hit_frac", "size"), hit_median=("hit_frac", "median"),
    hit_mean=("hit_frac", "mean"), lost_mean=("lost_prefill_s", "mean"), lost_p90=("lost_prefill_s", lambda x: x.quantile(.9)),
    call_mean=("call_s", "mean")).reset_index()
g.to_csv(HERE / "e1_summary.csv", index=False)
t = d[d.kind == "switch"].groupby(["src", "schema", "context", "to"]).lost_prefill_s.mean().reset_index()
t.to_csv(HERE / "e1_switch_by_target.csv", index=False)
print(g.round(3).to_string()); print(t.round(3).to_string())
