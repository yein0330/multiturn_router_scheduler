"""followup_v2 analysis (CPU only).

    python v2_analyze.py main     -> results/analysis_main/
    python v2_analyze.py branch   -> results/analysis_branch/
Session = resampling unit (bootstrap stratified by scenario, 2000 draws; descriptive only; sessions share
templates, so intervals understate uncertainty). Labels (difficulty/scenario) used for analysis only.
"""
import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
SESS = {json.loads(l)["session_id"]: json.loads(l) for l in open(ROOT / "data/eval/eval.jsonl")}
SIX = ["AAAAA", "AAAAB", "AAABB", "AABBB", "ABBBB", "BBBBB"]
SWITCH_STEP = {p: p.find("B") if "B" in p else 5 for p in SIX}
OLD = {"short": ROOT / "results/evalA", "long": ROOT / "results/evalB_long_retain"}


def jl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def sess_row(s, src, ctx, T):
    ts = T.get((s["session_id"], s["policy"], ctx), [])
    r = {"session_id": s["session_id"], "scenario": SESS[s["session_id"]]["scenario"], "policy": s["policy"],
         "context": ctx, "src": src, "step_pass": s["step_pass_rate"], "final_pass": float(s["final_pass"]),
         "all_pass": float(s["all_pass"]), "latency_s": s["latency_s"], "n_turns": s["n_turns"],
         "router_calls": s["router_calls"], "switches": s["switches"],
         "step_vec": "".join("1" if e["pass"] else "0" for e in s["steps"])}
    for m in "AB":
        mt = [t for t in ts if t["chosen"] == m]
        r[f"{m}_ttft_s"] = sum((t["call"]["ttft_s"] or t["call"]["total_s"]) for t in mt)
        r[f"{m}_gen_s"] = sum(t["call"]["total_s"] - (t["call"]["ttft_s"] or t["call"]["total_s"]) for t in mt)
        r[f"{m}_gen_tokens"] = sum(t["call"]["completion_tokens"] or 0 for t in mt)
        r[f"{m}_calls"] = len(mt)
    r["hit_tokens"] = sum(t["call"]["cached_tokens"] or 0 for t in ts)
    r["miss_tokens"] = sum(t["call"]["prompt_tokens"] - (t["call"]["cached_tokens"] or 0) for t in ts)
    r["verifier_fail"] = sum(1 for t in ts if t["verifier"] and t["verifier"]["status"] != "PASS")
    r["retries"] = sum(1 for t in ts if t["attempt"] > 0)
    r["hit_mismatch"] = sum(t["predicted_hit"] != t["call"]["cached_tokens"] for t in ts)
    r["truncated"] = sum(t["call"]["finish_reason"] == "length" for t in ts)
    return r


def turns_index(rows):
    T = collections.defaultdict(list)
    for t in rows:
        T[(t["session_id"], t["policy"], t["context"])].append(t)
    return T


def load_main():
    dec = json.load(open(V2 / "results/baseline_decision.json"))
    rows = []
    newS = jl(V2 / "results/main/sessions.jsonl")
    newT = turns_index(jl(V2 / "results/main/turns.jsonl"))
    for s in newS:
        rows.append(sess_row(s, "v2_main", s["context"], newT))
    for ctx, d in OLD.items():
        T = collections.defaultdict(list)
        for t in jl(d / "turns.jsonl"):
            T[(t["session_id"], t["policy"], ctx)].append(t)
        for s in jl(d / "sessions.jsonl"):
            pol = s["policy"]
            if pol in ("P1", "P2") and dec["reuse_old_baselines"]:
                r = sess_row(s, "reused_" + pol, ctx, T)
                r["policy"] = "AAAAA" if pol == "P1" else "BBBBB"
                rows.append(r)
            elif pol in ("P4", "P6"):
                rows.append(sess_row(s, "existing", ctx, T))
    T10 = collections.defaultdict(list)
    for t in jl(V2.parent / "followup_v1/results/newpol/turns.jsonl"):
        T10[(t["session_id"], t["policy"], "short")].append(t)
    for s in jl(V2.parent / "followup_v1/results/newpol/sessions.jsonl"):
        if s["policy"] == "P10":
            rows.append(sess_row(s, "followup_v1", "short", T10))
    return pd.DataFrame(rows), dec


def strat_boot(diff_by_session, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    by_sc = collections.defaultdict(list)
    for sid, v in diff_by_session.items():
        by_sc[SESS[sid]["scenario"]].append(v)
    arrs = [np.array(v) for v in by_sc.values()]
    means = []
    tot = sum(len(a) for a in arrs)
    for _ in range(n):
        means.append(sum(rng.choice(a, len(a)).sum() for a in arrs) / tot)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze_main():
    out = V2 / "results/analysis_main"
    out.mkdir(parents=True, exist_ok=True)
    df, dec = load_main()
    df.to_csv(out / "sessions_all.csv", index=False)
    audit = {"rows_by_src": df.groupby(["src", "context"]).size().to_dict(),
             "v2_hit_mismatch": int(df[df.src == "v2_main"].hit_mismatch.sum()),
             "v2_truncated": int(df[df.src == "v2_main"].truncated.sum()),
             "v2_runs": int((df.src == "v2_main").sum()),
             "v2_unique": int(df[df.src == "v2_main"][["session_id", "policy", "context"]].drop_duplicates().shape[0]),
             "baseline_reuse": dec["reuse_old_baselines"]}
    audit["rows_by_src"] = {f"{k[0]}|{k[1]}": v for k, v in audit["rows_by_src"].items()}
    json.dump(audit, open(out / "audit.json", "w"), indent=1)
    cols = ["step_pass", "final_pass", "all_pass", "latency_s", "A_ttft_s", "A_gen_s", "A_gen_tokens", "B_ttft_s",
            "B_gen_s", "B_gen_tokens", "hit_tokens", "miss_tokens", "verifier_fail", "retries", "router_calls"]
    summ = df.groupby(["context", "policy"])[cols].mean()
    summ["latency_median_s"] = df.groupby(["context", "policy"]).latency_s.median()
    summ["n"] = df.groupby(["context", "policy"]).size()
    summ = summ.reset_index()
    summ.to_csv(out / "summary.csv", index=False)
    sc = df.pivot_table(index=["context", "policy"], columns="scenario", values="step_pass").round(3)
    sc.to_csv(out / "step_pass_by_scenario.csv")
    scl = df.pivot_table(index=["context", "policy"], columns="scenario", values="latency_s").round(2)
    scl.to_csv(out / "latency_by_scenario.csv")
    # paired comparisons with stratified session bootstrap
    pr = []
    for ctx in ("short", "long"):
        d = df[df.context == ctx]
        piv = {m: d.pivot_table(index="session_id", columns="policy", values=m)
               for m in ("step_pass", "final_pass", "latency_s")}
        for a in SIX[1:-1] + ["P4", "P6", "P10"]:
            for b in ("BBBBB", "AAAAA", "P4"):
                if a == b or a not in piv["step_pass"] or b not in piv["step_pass"]:
                    continue
                rec = {"context": ctx, "a": a, "b": b}
                for m in ("step_pass", "final_pass", "latency_s"):
                    dd = (piv[m][a] - piv[m][b]).dropna()
                    rec["n"] = len(dd)
                    rec[f"d_{m}"] = dd.mean()
                    rec[f"d_{m}_ci"] = strat_boot(dd.to_dict())
                dq = (piv["step_pass"][a] - piv["step_pass"][b]).dropna()
                rec["n_better"], rec["n_worse"] = int((dq > 0).sum()), int((dq < 0).sum())
                pr.append(rec)
    pr = pd.DataFrame(pr)
    pr.to_csv(out / "paired.csv", index=False)
    # per-session ex-post best within the six-policy set (no weighted score)
    best = []
    for ctx in ("short", "long"):
        d = df[(df.context == ctx) & df.policy.isin(SIX)]
        for sid, g in d.groupby("session_id"):
            if len(g) < 6:
                continue
            pts = g[["policy", "step_pass", "latency_s"]].values
            pareto = [p for p, q, l in pts if not any((q2 >= q and l2 <= l) and (q2 > q or l2 < l) for _, q2, l2 in pts)]
            top_q = g.step_pass.max()
            lex = g[g.step_pass == top_q].sort_values("latency_s").iloc[0].policy
            best.append({"context": ctx, "session_id": sid, "scenario": SESS[sid]["scenario"],
                         "pattern": SESS[sid]["pattern"], "pareto_set": ",".join(pareto),
                         "max_quality": top_q, "lexicographic_best": lex,
                         "n_policies_at_max_quality": int((g.step_pass == top_q).sum())})
    best = pd.DataFrame(best)
    best.to_csv(out / "ex_post_best_within_six.csv", index=False)
    lexc = best.groupby(["context", "lexicographic_best"]).size().unstack(fill_value=0)
    lexc.to_csv(out / "ex_post_best_within_six_counts.csv")
    par = {ctx: collections.Counter(p for s in best[best.context == ctx].pareto_set for p in s.split(","))
           for ctx in ("short", "long")}
    json.dump(par, open(out / "pareto_membership_counts.json", "w"), indent=1)
    # recoverable vs unrecoverable: compare each switch policy with AAAAA on steps >= switch step
    rec_rows = []
    for ctx in ("short", "long"):
        d = df[df.context == ctx].set_index(["session_id", "policy"])
        for p in SIX[1:5]:
            k = SWITCH_STEP[p]
            for sid in SESS:
                if (sid, p) not in d.index or (sid, "AAAAA") not in d.index or (sid, "BBBBB") not in d.index:
                    continue
                va, vp, vb = d.loc[(sid, "AAAAA"), "step_vec"], d.loc[(sid, p), "step_vec"], d.loc[(sid, "BBBBB"), "step_vec"]
                pre_err = "0" in va[:k]
                rec_rows.append({"context": ctx, "policy": p, "switch_step": k, "session_id": sid,
                                 "scenario": SESS[sid]["scenario"], "pre_switch_error_on_A": pre_err,
                                 "after_pass_switch": vp[k:].count("1") / (5 - k),
                                 "after_pass_AAAAA": va[k:].count("1") / (5 - k),
                                 "after_pass_BBBBB": vb[k:].count("1") / (5 - k),
                                 "final_switch": vp[-1] == "1", "final_AAAAA": va[-1] == "1",
                                 "final_BBBBB": vb[-1] == "1"})
    rr = pd.DataFrame(rec_rows)
    rr.to_csv(out / "recovery_by_session.csv", index=False)
    rs = rr.groupby(["context", "policy", "pre_switch_error_on_A"]).agg(
        n=("session_id", "count"), after_pass_switch=("after_pass_switch", "mean"),
        after_pass_AAAAA=("after_pass_AAAAA", "mean"), after_pass_BBBBB=("after_pass_BBBBB", "mean"),
        final_switch=("final_switch", "mean"), final_AAAAA=("final_AAAAA", "mean"),
        final_BBBBB=("final_BBBBB", "mean")).reset_index()
    rs.to_csv(out / "recovery_summary.csv", index=False)
    # figures
    col = {"AAAAA": "#9ecae1", "AAAAB": "#6baed6", "AAABB": "#4292c6", "AABBB": "#2171b5", "ABBBB": "#08519c",
           "BBBBB": "#08306b", "P4": "#d62728", "P6": "#2ca02c", "P10": "#ff7f0e"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ctx in zip(axs, ("short", "long")):
        d = df[df.context == ctx]
        for p, g in d.groupby("policy"):
            q, l = g.step_pass.values, g.latency_s.values
            rng = np.random.default_rng(1)
            bq = [rng.choice(q, len(q)).mean() for _ in range(1000)]
            bl = [rng.choice(l, len(l)).mean() for _ in range(1000)]
            ax.errorbar(l.mean(), q.mean(), xerr=[[l.mean() - np.percentile(bl, 2.5)], [np.percentile(bl, 97.5) - l.mean()]],
                        yerr=[[q.mean() - np.percentile(bq, 2.5)], [np.percentile(bq, 97.5) - q.mean()]],
                        fmt="o", color=col.get(p, "gray"), capsize=2, ms=6)
            ax.annotate(p, (l.mean(), q.mean()), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_title(f"{ctx}/retain (session-bootstrap 95%, unstratified, descriptive)", fontsize=9)
        ax.set_xlabel("mean session latency (s)")
        ax.set_ylabel("mean step pass rate")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_quality_latency.png", dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ctx, ls in (("short", "-o"), ("long", "--s")):
        s = summ[(summ.context == ctx) & summ.policy.isin(SIX)].copy()
        s["k"] = s.policy.map(SWITCH_STEP)
        s = s.sort_values("k")
        axs[0].plot(s.k, s.step_pass, ls, label=ctx)
        axs[1].plot(s.k, s.final_pass, ls, label=ctx)
        axs[2].plot(s.k, s.latency_s, ls, label=ctx)
    for ax, t in zip(axs, ("step pass rate", "final step pass rate", "mean session latency (s)")):
        ax.set_xticks(range(6))
        ax.set_xticklabels(["0\nBBBBB", "1\nABBBB", "2\nAABBB", "3\nAAABB", "4\nAAAAB", "5\nAAAAA"], fontsize=7)
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("first B step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_switch_step.png", dpi=130)
    plt.close(fig)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(json.dumps(audit))
        print(summ.round(3).to_string())
        print(pr[["context", "a", "b", "n", "d_step_pass", "d_step_pass_ci", "d_final_pass", "d_latency_s",
                  "d_latency_s_ci", "n_better", "n_worse"]].round(3).to_string())
        print(sc.to_string())
        print(lexc.to_string())
        print(json.dumps(par))
        print(rs.round(3).to_string())


def analyze_branch():
    out = V2 / "results/analysis_branch"
    out.mkdir(parents=True, exist_ok=True)
    S = jl(V2 / "results/branch/sessions.jsonl")
    T = jl(V2 / "results/branch/turns.jsonl")
    pts = json.load(open(V2 / "configs/branch_points.json"))
    first = {}
    for t in T:
        k = (t["point"], t["path"], t["rep"])
        if k not in first or t["turn"] < first[k]["turn"]:
            first[k] = t
    old = {}
    for ctx, d in OLD.items():
        for s in jl(d / "sessions.jsonl"):
            if s["policy"] == "P1":
                old[(s["session_id"], ctx)] = s
    rows = []
    for s in S:
        k = (s["point"], s["path"], s["rep"])
        ft = first[k]
        pt = pts[s["point"]]
        p1 = old[(pt["session_id"], pt["context"])]
        pre_err = not all(e["pass"] for e in p1["steps"][: pt["step"]])
        rows.append({"point": s["point"], "session_id": pt["session_id"], "scenario": SESS[pt["session_id"]]["scenario"],
                     "context": pt["context"], "branch_step": pt["step"], "path": s["path"], "rep": s["rep"],
                     "reconstruct_ok": s["reconstruct_ok"], "prep_s": s["prep_s"],
                     "first_expected_hit": s["expected_first_hit"]["A" if s["path"] == "stayA" else "B"],
                     "first_hit": ft["call"]["cached_tokens"], "first_pred_hit": ft["predicted_hit"],
                     "next_call_pass": s["steps"][0]["pass"], "next_call_s": ft["turn_latency_s"],
                     "next_call_tokens": ft["call"]["completion_tokens"],
                     "remaining_pass": s["step_pass_rate"], "final_pass": s["final_pass"],
                     "remaining_latency_s": s["latency_s"], "n_turns": s["n_turns"],
                     "retries": sum(1 for t in T if (t["point"], t["path"], t["rep"]) == k and t["attempt"] > 0),
                     "gen_tokens": sum(t["call"]["completion_tokens"] or 0 for t in T if (t["point"], t["path"], t["rep"]) == k),
                     "hit_mismatch": sum(t["predicted_hit"] != t["call"]["cached_tokens"] for t in T
                                         if (t["point"], t["path"], t["rep"]) == k),
                     "text_sig": "|".join(t["assistant_text"][:0] + str(hash(t["assistant_text"])) for t in
                                          sorted([t for t in T if (t["point"], t["path"], t["rep"]) == k], key=lambda t: t["turn"])),
                     "pre_branch_error_on_A": pre_err})
    df = pd.DataFrame(rows)
    df.to_csv(out / "path_runs.csv", index=False)
    var = df.groupby(["point", "path"]).agg(n_reps=("rep", "count"), n_text_variants=("text_sig", "nunique"),
                                            n_quality_variants=("remaining_pass", "nunique")).reset_index()
    var.to_csv(out / "rep_variation.csv", index=False)
    agg = df.groupby(["point", "session_id", "scenario", "context", "branch_step", "path", "pre_branch_error_on_A"]).agg(
        next_call_pass=("next_call_pass", "mean"), next_call_s=("next_call_s", "median"),
        remaining_pass=("remaining_pass", "mean"), final_pass=("final_pass", "mean"),
        remaining_latency_s=("remaining_latency_s", "median"), lat_min=("remaining_latency_s", "min"),
        lat_max=("remaining_latency_s", "max"), gen_tokens=("gen_tokens", "median"),
        retries=("retries", "median")).reset_index()
    w = agg.pivot_table(index=["point", "session_id", "scenario", "context", "branch_step", "pre_branch_error_on_A"],
                        columns="path", values=["next_call_pass", "next_call_s", "remaining_pass", "final_pass",
                                                "remaining_latency_s", "gen_tokens", "retries"]).reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    w["d_remaining_pass_B_minus_A"] = w.remaining_pass_switchB - w.remaining_pass_stayA
    w["d_final_B_minus_A"] = w.final_pass_switchB - w.final_pass_stayA
    w["d_remaining_latency_B_minus_A"] = w.remaining_latency_s_switchB - w.remaining_latency_s_stayA
    w["d_next_latency_B_minus_A"] = w.next_call_s_switchB - w.next_call_s_stayA
    w.to_csv(out / "paired_points.csv", index=False)
    g = w.groupby(["context", "branch_step"]).agg(
        n=("point", "count"), remaining_pass_A=("remaining_pass_stayA", "mean"),
        remaining_pass_B=("remaining_pass_switchB", "mean"), final_A=("final_pass_stayA", "mean"),
        final_B=("final_pass_switchB", "mean"), d_latency=("d_remaining_latency_B_minus_A", "mean"),
        d_next_latency=("d_next_latency_B_minus_A", "mean"),
        n_B_better=("d_remaining_pass_B_minus_A", lambda x: int((x > 0).sum())),
        n_B_worse=("d_remaining_pass_B_minus_A", lambda x: int((x < 0).sum()))).reset_index()
    g.to_csv(out / "summary_by_context_step.csv", index=False)
    g2 = w.groupby(["context", "branch_step", "pre_branch_error_on_A"]).agg(
        n=("point", "count"), remaining_pass_A=("remaining_pass_stayA", "mean"),
        remaining_pass_B=("remaining_pass_switchB", "mean"), final_A=("final_pass_stayA", "mean"),
        final_B=("final_pass_switchB", "mean"), d_latency=("d_remaining_latency_B_minus_A", "mean")).reset_index()
    g2.to_csv(out / "summary_by_pre_error.csv", index=False)
    audit = {"path_runs": len(df), "expected": 240, "reconstruct_ok_all": bool(df.reconstruct_ok.all()),
             "first_hit_equals_expected": int((df.first_hit == df.first_expected_hit).sum()),
             "hit_mismatch_turns": int(df.hit_mismatch.sum()),
             "points_paths_with_text_variation": int((var.n_text_variants > 1).sum()),
             "points_paths_with_quality_variation": int((var.n_quality_variants > 1).sum()),
             "prep_s_median": float(df.prep_s.median())}
    json.dump(audit, open(out / "audit.json", "w"), indent=1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ctx in zip(axs, ("short", "long")):
        d = w[w.context == ctx]
        for k, mk in ((1, "o"), (3, "s")):
            dd = d[d.branch_step == k]
            ax.scatter(dd.d_remaining_latency_B_minus_A, dd.d_remaining_pass_B_minus_A + (k - 2) * 0.01, marker=mk,
                       label=f"branch at step {k}")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(f"{ctx}: switch to B now vs stay on A (same history)", fontsize=9)
        ax.set_xlabel("remaining-steps latency, B - A (s, median of 3)")
        ax.set_ylabel("remaining step pass, B - A")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_branch.png", dpi=130)
    plt.close(fig)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(json.dumps(audit))
        print(g.round(3).to_string())
        print(g2.round(3).to_string())
        print(w[["point", "session_id", "context", "branch_step", "pre_branch_error_on_A", "remaining_pass_stayA",
                 "remaining_pass_switchB", "final_pass_stayA", "final_pass_switchB",
                 "d_remaining_latency_B_minus_A", "d_next_latency_B_minus_A"]].round(3).to_string())


if __name__ == "__main__":
    {"main": analyze_main, "branch": analyze_branch}[sys.argv[1]]()
