"""Stage B analysis (CPU). -> followup_v4/results/stageB_analysis/
All quantities are from the branch point onward (post-state costs). Replicates are reproducibility checks only.
'Ex-post best' uses the realized outcomes of both branches: an optimistic diagnostic, not an adapter or a bound.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V4 = Path(__file__).resolve().parents[1]
ROOT = V4.parent
sys.path.insert(0, str(ROOT / "followup_v3/src"))
from profile_v3 import CostModels  # noqa: E402

OUT = V4 / "results/stageB_analysis"
MUS = [0.5, 1, 2, 4]
YHAT = {"A": 35.0, "B": 36.8}  # v3 dev-fixed output-length estimates (decision-time information)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = [json.loads(l) for l in open(V4 / "results/stageB/sessions.jsonl")]
    C = [json.loads(l) for l in open(V4 / "results/stageB/calls.jsonl")]
    RA = [json.loads(l) for l in open(V4 / "results/stageB/restore_audit.jsonl")]
    states = {s["state_id"]: s for s in json.load(open(V4 / "configs/stageB_states.json"))["states"]}
    A = pd.read_csv(V4 / "results/stageA_analysis/summary.csv")
    tref = {ctx: float(A[(A.context == ctx) & (A.schema == "T1") & (A.policy == "always_B")].latency_s.iloc[0])
            for ctx in ("short", "long")}
    cm = CostModels.load(ROOT / "followup_v3/results/stage01/profile/cost_models.json")
    by = collections.defaultdict(list)
    for c in C:
        by[(c["state_id"], c["branch"], c["rep"])].append(c)
    rows = []
    for s in S:
        cs = sorted(by[(s["state_id"], s["branch"], s["rep"])], key=lambda c: c["call"])
        f = cs[0]
        k = s["state_request"]
        reqs = s["requests"]
        prev_ok = all(s["prev_requests_correct"])
        r = {"state_id": s["state_id"], "context": s["context"], "category": s["category"], "branch": s["branch"],
             "rep": s["rep"], "first_model": s["first_model"], "rec": s["rec_at_state"], "session_id": s["session_id"],
             "state_call": s["state_call"], "state_request": k, "prev_all_correct": prev_ok,
             "d_first_s": f["call_latency_s"], "first_out_tokens": f["completion_tokens"], "first_prompt": f["prompt_tokens"],
             "first_hit": f["cached_tokens"], "J_s": s["latency_s"], "n_calls": s["n_calls"],
             "router_s": sum(c["router_s"] for c in cs), "model_s": sum(c["total_s"] for c in cs),
             "tool_s": sum(c["tool_exec_s"] for c in cs), "render_s": sum(c["render_s"] for c in cs),
             "A_calls": s["model_calls"]["A"], "B_calls": s["model_calls"]["B"], "switches": s["switches"],
             "tool_errors": sum(1 for c in cs for t in c["tools"] if t["error"]),
             "budget_exhausted": sum(not q["submitted"] for q in reqs[k:]),
             "cur_req_correct": float(reqs[k]["correct"]),
             "rem_q": float(np.mean([q["correct"] for q in reqs[k:]])), "S": float(s["S"]),
             "texts": [c["assistant_text"] for c in cs]}
        for rep_ in ("C1", "C2", "C3"):
            m = f["chosen"]
            r[f"{rep_}_first_yreal"] = cm.predict(rep_, m, f["prompt_tokens"], f["pred_hit"][m], f["completion_tokens"])
            r[f"{rep_}_first_yhat"] = cm.predict(rep_, m, f["prompt_tokens"], f["pred_hit"][m], YHAT[m])
        r["first_model_s"] = f["total_s"]
        rows.append(r)
    d = pd.DataFrame(rows)
    d.drop(columns=["texts"]).to_csv(OUT / "branch_runs.csv", index=False)
    # restore audit
    ra = pd.DataFrame(RA)
    audit = {"branch_runs": len(d), "states": int(d.state_id.nunique()), "prompt_hash_ok": int(ra.prompt_hash_ok.sum()),
             "first_hit_ok": int(ra.first_hit_ok.sum()), "restore_rows": len(ra), "prep_s_median": float(ra.prep_s.median()),
             "calls": len(C), "cache_mismatch_all_calls": sum(c["pred_hit"][c["chosen"]] != c["cached_tokens"] for c in C),
             "truncated": sum(c["finish"] == "length" for c in C), "http_errors": sum(bool(c["error"]) for c in C),
             "router_fallback": sum(bool(c["router"] and c["router"]["parse_error"]) for c in C)}
    # replicate reproducibility
    rep = []
    for (sid, br), g in d.groupby(["state_id", "branch"]):
        if g.rep.nunique() > 1:
            a, b = g[g.rep == 0].iloc[0], g[g.rep == 1].iloc[0]
            ta = next(x["texts"] for x in rows if x["state_id"] == sid and x["branch"] == br and x["rep"] == 0)
            tb = next(x["texts"] for x in rows if x["state_id"] == sid and x["branch"] == br and x["rep"] == 1)
            rep.append({"state_id": sid, "branch": br, "same_text": ta == tb, "same_S": a.S == b.S, "same_rem_q": a.rem_q == b.rem_q,
                        "J_rep0": a.J_s, "J_rep1": b.J_s, "J_ratio": b.J_s / a.J_s})
    rp = pd.DataFrame(rep)
    rp.to_csv(OUT / "replicates.csv", index=False)
    audit["replicates"] = {"pairs": len(rp), "same_text": int(rp.same_text.sum()) if len(rp) else 0,
                           "J_ratio_range": [float(rp.J_ratio.min()), float(rp.J_ratio.max())] if len(rp) else None}
    # paired per state (rep 0)
    d0 = d[d.rep == 0]
    f_ = d0[d0.branch == "follow"].set_index("state_id")
    v_ = d0[d0.branch == "deviate"].set_index("state_id")
    idx = f_.index.intersection(v_.index)
    P = pd.DataFrame({"context": f_.loc[idx, "context"], "category": f_.loc[idx, "category"],
                      "session_id": f_.loc[idx, "session_id"], "rec": f_.loc[idx, "rec"],
                      "follow_model": f_.loc[idx, "first_model"], "deviate_model": v_.loc[idx, "first_model"],
                      "prev_all_correct": f_.loc[idx, "prev_all_correct"],
                      "d_first": v_.loc[idx, "d_first_s"] - f_.loc[idx, "d_first_s"],
                      "dJ": v_.loc[idx, "J_s"] - f_.loc[idx, "J_s"],
                      "dS": v_.loc[idx, "S"] - f_.loc[idx, "S"],
                      "d_cur_req": v_.loc[idx, "cur_req_correct"] - f_.loc[idx, "cur_req_correct"],
                      "d_rem_q": v_.loc[idx, "rem_q"] - f_.loc[idx, "rem_q"],
                      "d_calls": v_.loc[idx, "n_calls"] - f_.loc[idx, "n_calls"],
                      "d_tool_errors": v_.loc[idx, "tool_errors"] - f_.loc[idx, "tool_errors"],
                      "d_B_calls": v_.loc[idx, "B_calls"] - f_.loc[idx, "B_calls"],
                      "d_switches": v_.loc[idx, "switches"] - f_.loc[idx, "switches"],
                      "follow_J": f_.loc[idx, "J_s"], "deviate_J": v_.loc[idx, "J_s"],
                      "follow_S": f_.loc[idx, "S"], "deviate_S": v_.loc[idx, "S"],
                      "follow_rem_q": f_.loc[idx, "rem_q"], "deviate_rem_q": v_.loc[idx, "rem_q"],
                      "C2_pred_d_first_yhat": v_.loc[idx, "C2_first_yhat"] - f_.loc[idx, "C2_first_yhat"],
                      "C2_pred_d_first_yreal": v_.loc[idx, "C2_first_yreal"] - f_.loc[idx, "C2_first_yreal"],
                      "measured_d_first_model": v_.loc[idx, "first_model_s"] - f_.loc[idx, "first_model_s"]})
    P["S_fixed_zero"] = ~P.prev_all_correct

    def case(r):
        if r.dS == 0 and r.d_rem_q == 0 and r.follow_rem_q == 0 and r.deviate_rem_q == 0:
            return "both_fail"
        if r.d_first < 0 and r.dJ > 0:
            return "first_cheaper_rest_costlier"
        if r.d_first > 0 and r.dJ < 0:
            return "first_costlier_rest_cheaper"
        if r.dJ < 0 and r.d_rem_q < 0:
            return "cheaper_but_worse"
        if r.d_rem_q == 0 and r.dS == 0:
            return "same_quality_cost_differs"
        return "other"
    P["case"] = P.apply(case, axis=1)
    P.to_csv(OUT / "paired_states.csv")
    # strategies over the same state set (rep 0)
    strat = []
    for ctx, g in P.groupby("context"):
        fj, dj = g.follow_J, g.deviate_J
        fs, ds = g.follow_S, g.deviate_S
        fq, dq = g.follow_rem_q, g.deviate_rem_q
        firstA = np.where(g.follow_model == "A", True, False)
        res = {"always_follow": (fj, fs, fq), "always_deviate": (dj, ds, dq),
               "first_always_A": (np.where(firstA, fj, dj), np.where(firstA, fs, ds), np.where(firstA, fq, dq)),
               "first_always_B": (np.where(~firstA, fj, dj), np.where(~firstA, fs, ds), np.where(~firstA, fq, dq))}
        # ex-post best: higher remaining quality, then lower J (quality-first lexicographic; diagnostic)
        pick_dev = (dq > fq) | ((dq == fq) & (dj < fj))
        res["ex_post_best_quality_first"] = (np.where(pick_dev, dj, fj), np.where(pick_dev, ds, fs), np.where(pick_dev, dq, fq))
        for name, (J, Sx, Q) in res.items():
            row = {"context": ctx, "strategy": name, "n_states": len(g), "J_mean": float(np.mean(J)),
                   "S_mean": float(np.mean(Sx)), "rem_q_mean": float(np.mean(Q))}
            for mu in MUS:
                row[f"U_S_mu{mu}"] = float(np.mean(mu * np.asarray(Sx) - np.asarray(J) / tref[ctx]))
                row[f"U_remq_mu{mu}"] = float(np.mean(mu * np.asarray(Q) - np.asarray(J) / tref[ctx]))
            strat.append(row)
        for mu in MUS:  # ex-post best under each mu (diagnostic)
            for lab, F, Dv in (("S", fs, ds), ("remq", fq, dq)):
                uf, ud = mu * F - fj / tref[ctx], mu * Dv - dj / tref[ctx]
                strat.append({"context": ctx, "strategy": f"ex_post_best_U_{lab}_mu{mu}", "n_states": len(g),
                              f"U_{lab}_mu{mu}": float(np.mean(np.maximum(uf, ud))),
                              "share_deviate_better": float(np.mean(ud > uf))})
    st = pd.DataFrame(strat)
    st.to_csv(OUT / "strategies.csv", index=False)
    # by category
    cat = P.groupby(["context", "category"]).agg(n=("dJ", "size"), d_first=("d_first", "mean"), dJ=("dJ", "mean"),
                                                   dS=("dS", "mean"), d_rem_q=("d_rem_q", "mean"), d_calls=("d_calls", "mean"),
                                                   deviate_better_q=("d_rem_q", lambda x: int((x > 0).sum())),
                                                   follow_better_q=("d_rem_q", lambda x: int((x < 0).sum())),
                                                   S_fixed_zero=("S_fixed_zero", "sum")).reset_index()
    cat.to_csv(OUT / "by_category.csv", index=False)
    # cost representations: first-call model time
    ce = []
    for ctx, g in d.groupby("context"):
        for rep_ in ("C1", "C2", "C3"):
            for v in ("yreal", "yhat"):
                e = (g[f"{rep_}_first_{v}"] - g.first_model_s).abs()
                ce.append({"context": ctx, "rep": rep_, "y": v, "mae_first_call_s": float(e.mean()), "n": len(g)})
    ce = pd.DataFrame(ce)
    ce.to_csv(OUT / "cost_repr_first_call.csv", index=False)
    audit["first_vs_remaining"] = {"corr_d_first_dJ": float(np.corrcoef(P.d_first, P.dJ)[0, 1]) if len(P) > 2 else None,
                                   "sign_agree_d_first_dJ": float((np.sign(P.d_first) == np.sign(P.dJ)).mean()),
                                   "C2_yhat_sign_vs_measured_first": float((np.sign(P.C2_pred_d_first_yhat) == np.sign(P.measured_d_first_model)).mean()),
                                   "C2_yhat_sign_vs_dJ": float((np.sign(P.C2_pred_d_first_yhat) == np.sign(P.dJ)).mean())}
    json.dump(audit, open(OUT / "audit.json", "w"), indent=1, default=float)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(json.dumps(audit, indent=1, default=float))
        print(cat.round(3).to_string())
        print(st[st.strategy.isin(["always_follow", "always_deviate", "first_always_A", "first_always_B",
                                   "ex_post_best_quality_first"])][["context", "strategy", "n_states", "J_mean", "S_mean",
                                                                     "rem_q_mean", "U_S_mu1", "U_remq_mu1", "U_remq_mu4"]].round(3).to_string())
        print(P.case.value_counts().to_string())
        print(ce.pivot_table(index=["context", "y"], columns="rep", values="mae_first_call_s").round(3))


if __name__ == "__main__":
    main()
