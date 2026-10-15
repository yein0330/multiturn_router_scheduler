"""Part 2 summary: P9/P10 (post-hoc, added after seeing stage A) vs existing P3/P5/P4/P6 from evalA,
paired by session; drift check = P4/P6 reruns vs the same sessions in evalA (not extra samples)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FU = Path(__file__).resolve().parents[1]
ROOT = FU.parent


def load(p):
    return [json.loads(l) for l in open(p)]


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    new = load(FU / "results/newpol/sessions.jsonl")
    newT = load(FU / "results/newpol/turns.jsonl")
    base = load(ROOT / "results/evalA/sessions.jsonl")
    rows = lambda S, tag: [{"session_id": s["session_id"], "scenario": s["scenario"], "policy": s["policy"], "src": tag,
                           "step_pass": s["step_pass_rate"], "final_pass": float(s["final_pass"]),
                           "all_pass": float(s["all_pass"]), "latency_s": s["latency_s"],
                           "router_calls": s["router_calls"], "switches": s["switches"],
                           "B_share": s["model_calls"]["B"] / s["n_turns"]} for s in S]
    nd, bd = pd.DataFrame(rows(new, "followup")), pd.DataFrame(rows(base, "evalA"))
    audit = {"rows": len(new), "unique": nd[["session_id", "policy"]].drop_duplicates().shape[0],
             "hit_mismatch": sum(t["predicted_hit"] != t["call"]["cached_tokens"] for t in newT),
             "truncated": sum(t["call"]["finish_reason"] == "length" for t in newT),
             "router_parse_errors": sum(bool(t["router"] and t["router"]["parse_error"]) for t in newT),
             "expected": 50 * 2 + 10 * 2}
    # drift: reruns vs evalA (same session, same policy)
    rr = nd[nd.policy.isin(["P4", "P6"])].merge(bd, on=["session_id", "policy"], suffixes=("_rerun", "_evalA"))
    rr["same_quality"] = rr.step_pass_rerun == rr.step_pass_evalA
    rr["lat_ratio"] = rr.latency_s_rerun / rr.latency_s_evalA
    texts_same = None
    try:  # compare outputs text by text for the reruns
        A = {}
        for t in load(ROOT / "results/evalA/turns.jsonl"):
            A[(t["session_id"], t["policy"], t["turn"])] = t["assistant_text"]
        diff = [t for t in newT if t["policy"] in ("P4", "P6") and A.get((t["session_id"], t["policy"], t["turn"])) != t["assistant_text"]]
        texts_same = len(diff) == 0
        audit["rerun_turns_text_diff"] = len(diff)
    except FileNotFoundError:
        pass
    rr.to_csv(out / "drift_rerun_vs_evalA.csv", index=False)
    audit["drift_same_quality"] = int(rr.same_quality.sum()); audit["drift_n"] = len(rr)
    audit["drift_latency_ratio_median"] = float(rr.lat_ratio.median()) if len(rr) else None
    audit["drift_latency_ratio_range"] = [float(rr.lat_ratio.min()), float(rr.lat_ratio.max())] if len(rr) else None
    # main table: P9/P10 (followup) + P3/P5/P4/P6/P1/P2 (evalA baseline)
    allp = pd.concat([nd[nd.policy.isin(["P9", "P10"])], bd[bd.policy.isin(["P1", "P2", "P3", "P4", "P5", "P6"])]])
    allp.to_csv(out / "sessions_combined.csv", index=False)
    summ = allp.groupby("policy").agg(n=("session_id", "count"), step_pass=("step_pass", "mean"),
                                      final_pass=("final_pass", "mean"), all_pass=("all_pass", "mean"),
                                      latency_s=("latency_s", "mean"), latency_med=("latency_s", "median"),
                                      router_calls=("router_calls", "mean"), switches=("switches", "mean"),
                                      B_share=("B_share", "mean")).reset_index()
    summ.to_csv(out / "summary_overall.csv", index=False)
    sc = allp.pivot_table(index="policy", columns="scenario", values="step_pass").round(3)
    sc.to_csv(out / "step_pass_by_scenario.csv")
    scl = allp.pivot_table(index="policy", columns="scenario", values="latency_s").round(2)
    scl.to_csv(out / "latency_by_scenario.csv")
    piv_q = allp.pivot_table(index="session_id", columns="policy", values="step_pass")
    piv_l = allp.pivot_table(index="session_id", columns="policy", values="latency_s")
    pr = []
    rng = np.random.default_rng(0)
    for a, b in [("P9", "P5"), ("P9", "P3"), ("P9", "P4"), ("P10", "P6"), ("P10", "P4"), ("P10", "P5"), ("P10", "P9")]:
        dq = (piv_q[a] - piv_q[b]).dropna(); dl = (piv_l[a] - piv_l[b]).dropna()
        bs = rng.choice(dq.values, (2000, len(dq))).mean(1)
        pr.append({"a": a, "b": b, "n": len(dq), "d_step_pass": dq.mean(),
                   "ci95": (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
                   "d_latency_s": dl.mean(), "n_better": int((dq > 0).sum()), "n_worse": int((dq < 0).sum())})
    pd.DataFrame(pr).to_csv(out / "paired.csv", index=False)
    reasons = pd.Series([t.get("route_reason") for t in newT if t["router"] and t["policy"] == "P10"]).value_counts()
    reasons.to_csv(out / "p10_route_reasons.csv")
    json.dump(audit, open(out / "audit.json", "w"), indent=1)
    with pd.option_context("display.width", 220):
        print(json.dumps(audit)); print(summ.round(3).to_string()); print(sc.to_string()); print(scl.to_string())
        print(pd.DataFrame(pr).round(3).to_string()); print(reasons)


if __name__ == "__main__":
    main(sys.argv[1])
