"""E3. Noisy vs silent failures of always-A (T1), and the cost of repeated calls (T1, T0)."""
import collections
import pandas as pd
from load import load_runs, index, has_repeat, first_repeat_call_index, tool_sig, HERE

S, C = load_runs()
run, req = index(S, C)
rows = []
for k, cs in req.items():
    src, sid, ctx, sch, pol, r = k
    if src != "v4A" or pol != "always_A":
        continue
    a = run[k[:5]]["requests"][r]["correct"]
    b = run[(src, sid, ctx, sch, "always_B")]["requests"][r]["correct"]
    rep = has_repeat(cs)
    rows.append({"schema": sch, "context": ctx, "session_id": sid, "request": r, "A_correct": a, "B_correct": b,
                 "repeat": rep, "first_repeat_call": first_repeat_call_index(cs),
                 "type": "A_enough" if a else ("noisy_fail" if rep else "silent_fail")})
d = pd.DataFrame(rows)
d.to_csv(HERE / "e3_requests.csv", index=False)
t1 = d[d.schema == "T1"]
print("T1 types:", t1.type.value_counts().to_dict(), "| A-only correct:", int((t1.A_correct & ~t1.B_correct).sum()))
print("T1 repeat x correct:", t1.groupby(["repeat", "A_correct"]).size().to_dict())
print("T1 B-needed by type:", t1[~t1.A_correct & t1.B_correct].type.value_counts().to_dict())
print("T1 first repeat call index (0-based) distribution:", t1.first_repeat_call.dropna().astype(int).value_counts().sort_index().to_dict())
# share of A calls / A latency spent on repeated calls
share = []
for sch in ("T0", "T1"):
    n = nrep = lat = latrep = 0
    for k, cs in req.items():
        if k[0] != "v4A" or k[3] != sch or k[4] != "always_A":
            continue
        prev = None
        for c in cs:
            sig = tool_sig(c)
            is_rep = bool(sig) and prev is not None and sig[0] == prev
            n += 1; lat += c["call_latency_s"]
            if is_rep:
                nrep += 1; latrep += c["call_latency_s"]
            prev = sig[-1] if sig else prev
    share.append({"schema": sch, "A_calls": n, "repeat_calls": nrep, "call_share": nrep / n, "latency_share": latrep / lat})
pd.DataFrame(share).to_csv(HERE / "e3_repeat_share.csv", index=False)
print(pd.DataFrame(share).round(3).to_string())
