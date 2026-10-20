"""E5. Request-level vs session-level ex-post A/B selection (T1) and router first-call diagnosis.
Ex-post selections use realized outcomes (optimistic diagnostics); request level additionally assumes independence."""
import collections
import pandas as pd
from load import load_runs, index, difficulty, SESS, HERE

S, C = load_runs()
run, req = index(S, C)
out, diag, diff = [], collections.Counter(), collections.defaultdict(lambda: [0, 0])
for ctx in ("short", "long"):
    sids = sorted({k[1] for k in run if k[0] == "v4A" and k[2] == ctx})
    tot = collections.defaultdict(float); n = 0
    for sid in sids:
        kA, kB, kR = (("v4A", sid, ctx, "T1", p) for p in ("always_A", "always_B", "follow_R1"))
        qa_s, qb_s = run[kA]["q"], run[kB]["q"]
        la_s, lb_s = run[kA]["latency_s"], run[kB]["latency_s"]
        pick = "A" if qa_s >= qb_s else "B"
        tot["sess_q"] += qa_s if pick == "A" else qb_s; tot["sess_lat"] += la_s if pick == "A" else lb_s
        tot["B_q"] += qb_s; tot["B_lat"] += lb_s; tot["R1_q"] += run[kR]["q"]; tot["R1_lat"] += run[kR]["latency_s"]
        for r in range(len(SESS[sid]["requests"])):
            a, b = run[kA]["requests"][r]["correct"], run[kB]["requests"][r]["correct"]
            la = sum(c["call_latency_s"] for c in req[kA + (r,)]); lb = sum(c["call_latency_s"] for c in req[kB + (r,)])
            useA = a >= b
            tot["req_q"] += a if useA else b; tot["req_lat"] += la if useA else lb; tot["req_A_share"] += useA; n += 1
            first = req[kR + (r,)][0]["chosen"]
            if (not a) and b:
                diag[(ctx, "needB", first)] += 1
            if a:
                diag[(ctx, "A_enough", first)] += 1
            dk = difficulty(SESS[sid]["requests"][r]["kind"])
            diff[(dk, "A")][0] += a; diff[(dk, "A")][1] += 1; diff[(dk, "B")][0] += b; diff[(dk, "B")][1] += 1
    ns = len(sids)
    out.append({"context": ctx, "B_q": tot["B_q"] / ns, "B_lat": tot["B_lat"] / ns, "R1_q": tot["R1_q"] / ns,
                "R1_lat": tot["R1_lat"] / ns, "session_expost_q": tot["sess_q"] / ns, "session_expost_lat": tot["sess_lat"] / ns,
                "request_expost_q": tot["req_q"] / n, "request_expost_lat_per_session": tot["req_lat"] / ns,
                "request_expost_A_share": tot["req_A_share"] / n})
o = pd.DataFrame(out)
o["session_expost_lat_vs_B"] = o.session_expost_lat / o.B_lat
o["request_expost_lat_vs_B"] = o.request_expost_lat_per_session / o.B_lat
o.to_csv(HERE / "e5_expost.csv", index=False)
pd.DataFrame([{"context": k[0], "label": k[1], "R1_first_choice": k[2], "n": v} for k, v in sorted(diag.items())]).to_csv(HERE / "e5_router_first_call.csv", index=False)
pd.DataFrame([{"difficulty": k[0], "model": k[1], "correct": v[0], "n": v[1], "rate": v[0] / v[1]} for k, v in sorted(diff.items())]).to_csv(HERE / "e5_difficulty.csv", index=False)
print(o.round(3).T.to_string()); print(dict(diag)); print({k: round(v[0] / v[1], 2) for k, v in diff.items()})
