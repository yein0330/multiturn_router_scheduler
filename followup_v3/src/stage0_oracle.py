"""Stage 0-1: ex-post selection headroom within executed policy sets (design.md, stage 0).

    python stage0_oracle.py
Outputs -> followup_v3/results/stage0/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

V3 = Path(__file__).resolve().parents[1]
ROOT = V3.parent
OUT = V3 / "results/stage0"
SIX = ["AAAAA", "AAAAB", "AAABB", "AABBB", "ABBBB", "BBBBB"]
MUS = [0.5, 1, 2, 4]


def sets_for(ctx):
    plus = SIX + ["P4", "P6"] + (["P10"] if ctx == "short" else [])
    return {"F2": ["AAAAA", "BBBBB"], "F6": SIX, "F6plus": plus}


def per_session_pick(piv_q, piv_l, piv_s, cand, mu=None, tref=None):
    """Return per-session chosen policy, latency, q, S."""
    rows = []
    for sid in piv_q.index:
        q, l, s = piv_q.loc[sid, cand], piv_l.loc[sid, cand], piv_s.loc[sid, cand]
        if mu is None:
            ok = q[q >= piv_q.loc[sid, "BBBBB"] - 1e-12].index
            p = l[ok].idxmin()
        else:
            p = (mu * q - l / tref).idxmax()
        rows.append((sid, p, l[p], q[p], s[p]))
    return pd.DataFrame(rows, columns=["session_id", "chosen", "latency", "q", "S"]).set_index("session_id")


def crossfit(piv_q, piv_l, piv_s, scen, rng_seed, mu=None, tref=None, sessions=None):
    """One random 5/5 split per scenario; pick one schedule on the selection half, evaluate on the other."""
    rng = np.random.default_rng(rng_seed)
    sessions = list(piv_q.index) if sessions is None else sessions
    res = []
    for sc in sorted(set(scen[s] for s in sessions)):
        ss = [s for s in sessions if scen[s] == sc]
        perm = rng.permutation(len(ss))
        sel = [ss[i] for i in perm[: len(ss) // 2]]
        ev = [ss[i] for i in perm[len(ss) // 2:]]
        mq, ml = piv_q.loc[sel, SIX].mean(), piv_l.loc[sel, SIX].mean()
        if mu is None:
            ok = mq[mq >= mq["BBBBB"] - 1e-12].index
            p = ml[ok].idxmin()
        else:
            p = (mu * mq - ml / tref).idxmax()
        for s in ev:
            res.append((s, sc, p, piv_l.loc[s, p], piv_q.loc[s, p], piv_s.loc[s, p]))
    return pd.DataFrame(res, columns=["session_id", "scenario", "chosen", "latency", "q", "S"])


def strat_resample(sessions, scen, rng):
    out = []
    for sc in sorted(set(scen.values())):
        ss = [s for s in sessions if scen[s] == sc]
        out += list(rng.choice(ss, len(ss)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V3.parent / "followup_v2/results/analysis_main/sessions_all.csv")
    summary, by_sc, picks_all = [], [], []
    defn = {"quality_constraint": "per-session request pass rate q_i(pi) >= q_i(BBBBB)",
            "session_success": "S = all steps pass (reported separately)",
            "candidate_sets": {c: sets_for(c) for c in ("short", "long")},
            "information_access": "ex-post: uses each session's realized outcomes of every executed policy; "
                                  "not deployable; within executed-path set only (no path splicing)",
            "T_ref": "mean latency of BBBBB on the same eval condition (post-hoc normalization; v3 policies "
                     "will use a dev-fixed T_ref)", "mus": MUS,
            "crossfit": "scenario-informed diagnostic baseline: per scenario 5/5 session split, select one of six "
                        "schedules on half, evaluate on the other half; 200 splits (not independent samples)",
            "bootstrap": "scenario-stratified session bootstrap, 2000; cross-fit re-run inside each resample"}
    json.dump(defn, open(OUT / "headroom_definition.json", "w"), indent=1)
    for ctx in ("short", "long"):
        d = df[df.context == ctx]
        piv_q = d.pivot_table(index="session_id", columns="policy", values="step_pass")
        piv_l = d.pivot_table(index="session_id", columns="policy", values="latency_s")
        piv_s = d.pivot_table(index="session_id", columns="policy", values="all_pass")
        scen = d.groupby("session_id").scenario.first().to_dict()
        tref = piv_l["BBBBB"].mean()
        sessions = list(piv_q.index)
        crit = [("constraint", None)] + [(f"mu={m}", m) for m in MUS]
        for cname, mu in crit:
            for sname, cand in list(sets_for(ctx).items()) + [("crossfit_F6", None)]:
                if cand is not None:
                    pk = per_session_pick(piv_q, piv_l, piv_s, cand, mu, tref)
                    pk["scenario"] = pk.index.map(scen)
                    val = (pk.latency.mean(), pk.q.mean(), pk.S.mean())
                    boots = []
                    rng = np.random.default_rng(0)
                    for _ in range(2000):
                        rs = strat_resample(sessions, scen, rng)
                        boots.append((pk.loc[rs, "latency"].mean(), pk.loc[rs, "q"].mean()))
                    picks = pk.assign(context=ctx, criterion=cname, set=sname)
                    picks_all.append(picks.reset_index())
                    for sc, g in pk.groupby("scenario"):
                        by_sc.append({"context": ctx, "criterion": cname, "set": sname, "scenario": sc,
                                      "latency": g.latency.mean(), "q": g.q.mean(), "S": g.S.mean(),
                                      "choice_counts": g.chosen.value_counts().to_dict()})
                else:
                    reps = [crossfit(piv_q, piv_l, piv_s, scen, i, mu, tref) for i in range(200)]
                    allr = pd.concat(reps)
                    val = (allr.latency.mean(), allr.q.mean(), allr.S.mean())
                    boots = []
                    rng = np.random.default_rng(1)
                    for b in range(300):  # selection re-run inside each resample (300 to bound cost)
                        rs = strat_resample(sessions, scen, rng)
                        # duplicate sessions get unique ids so the split treats them as separate units
                        uid = [f"{s}#{k}" for k, s in enumerate(rs)]
                        q2 = piv_q.loc[rs].set_axis(uid); l2 = piv_l.loc[rs].set_axis(uid); s2 = piv_s.loc[rs].set_axis(uid)
                        sc2 = {u: scen[s] for u, s in zip(uid, rs)}
                        r = pd.concat([crossfit(q2, l2, s2, sc2, 1000 + b * 7 + i, mu, tref) for i in range(5)])
                        boots.append((r.latency.mean(), r.q.mean()))
                    for sc, g in allr.groupby("scenario"):
                        by_sc.append({"context": ctx, "criterion": cname, "set": sname, "scenario": sc,
                                      "latency": g.latency.mean(), "q": g.q.mean(), "S": g.S.mean(),
                                      "choice_counts": g.chosen.value_counts().to_dict()})
                b = np.array(boots)
                summary.append({"context": ctx, "criterion": cname, "set": sname, "T_ref_post_hoc": tref,
                                "latency": val[0], "latency_ci": (float(np.percentile(b[:, 0], 2.5)), float(np.percentile(b[:, 0], 97.5))),
                                "q": val[1], "q_ci": (float(np.percentile(b[:, 1], 2.5)), float(np.percentile(b[:, 1], 97.5))),
                                "S": val[2], "latency_vs_BBBBB": val[0] / tref,
                                "BBBBB_q": piv_q["BBBBB"].mean(), "BBBBB_S": piv_s["BBBBB"].mean(),
                                "note": "optimistic ex-post selection" if cand is not None else
                                        "scenario-informed diagnostic baseline (not a policy)"})
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "oracle_summary.csv", index=False)
    pd.DataFrame(by_sc).to_csv(OUT / "oracle_by_scenario.csv", index=False)
    pd.concat(picks_all).to_csv(OUT / "oracle_picks_by_session.csv", index=False)
    # branch per remaining stage
    bp = pd.read_csv(V3.parent / "followup_v2/results/analysis_branch/paired_points.csv")
    bp["remaining_steps"] = 5 - bp.branch_step
    for p in ("stayA", "switchB"):
        bp[f"latency_per_remaining_step_{p}"] = bp[f"remaining_latency_s_{p}"] / bp.remaining_steps
    bp["d_latency_per_remaining_step"] = bp.d_remaining_latency_B_minus_A / bp.remaining_steps
    bp.to_csv(OUT / "branch_per_remaining_stage.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(s[["context", "criterion", "set", "latency", "latency_ci", "q", "q_ci", "S", "latency_vs_BBBBB",
                 "BBBBB_q", "BBBBB_S"]].round(3).to_string())
        print(bp.groupby(["context", "branch_step"])[["d_latency_per_remaining_step"]].mean().round(3))


if __name__ == "__main__":
    main()
