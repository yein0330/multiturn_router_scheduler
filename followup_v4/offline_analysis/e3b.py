"""E3b. Optimistic simulation of 'escalate to B at the first repeated call' (T1).
Assumptions (optimistic): the B outcome/latency of the same request in the always-B run is used as if B started
from the original history; requests are treated as independent."""
import pandas as pd
from load import load_runs, index, first_repeat_call_index, HERE

S, C = load_runs()
run, req = index(S, C)
rows = []
for k, cs in req.items():
    src, sid, ctx, sch, pol, r = k
    if src != "v4A" or sch != "T1" or pol != "always_A":
        continue
    kb, kr = (src, sid, ctx, sch, "always_B", r), (src, sid, ctx, sch, "follow_R1", r)
    la, lb, lr = (sum(c["call_latency_s"] for c in req[x]) for x in (k, kb, kr))
    qa = run[k[:5]]["requests"][r]["correct"]; qb = run[kb[:5]]["requests"][r]["correct"]; qr = run[kr[:5]]["requests"][r]["correct"]
    i = first_repeat_call_index(cs)
    if i is None:
        lrule, qrule = la, qa
    else:
        lrule, qrule = sum(c["call_latency_s"] for c in cs[: i + 1]) + lb, qb
    rows.append({"context": ctx, "always_A": (qa, la), "rule": (qrule, lrule), "follow_R1": (qr, lr), "always_B": (qb, lb)})
out = []
for ctx in ("short", "long"):
    g = [x for x in rows if x["context"] == ctx]
    for p in ("always_A", "rule", "follow_R1", "always_B"):
        out.append({"context": ctx, "policy": p, "accuracy": sum(x[p][0] for x in g) / len(g),
                    "latency_per_request_s": sum(x[p][1] for x in g) / len(g), "n": len(g)})
o = pd.DataFrame(out); o.to_csv(HERE / "e3b_rule_sim.csv", index=False)
print(o.round(3).to_string())
