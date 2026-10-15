"""Tables and figures from results/<run>/{sessions,turns}.jsonl.

    python -m mrp.analyze --runs evalA[,evalB_long_retain,...] --out results/analysis_<name>

All numbers are re-derived from the logs. Paired comparisons use the same session under two
policies; intervals are a session-level bootstrap (descriptive only, not an inference claim).
"""
import argparse
import collections
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
POL = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
PAIRS = [("P3", "P4"), ("P5", "P4"), ("P6", "P4"), ("P7", "P4"), ("P8", "P6"), ("P8", "P7"),
         ("P4", "P1"), ("P4", "P2"), ("P6", "P2"), ("P8", "P2"), ("P2", "P1")]
COL = {"P1": "#7f7f7f", "P2": "#000000", "P3": "#1f77b4", "P4": "#d62728", "P5": "#9467bd",
       "P6": "#2ca02c", "P7": "#ff7f0e", "P8": "#17becf"}


def load(runs):
    S, T = [], []
    for r in runs:
        d = ROOT / "results" / r
        S += [json.loads(l) for l in open(d / "sessions.jsonl")]
        T += [json.loads(l) for l in open(d / "turns.jsonl")]
    # P1/P2 never switch, so evict == retain for them (design.md §9): reuse retain runs, flagged.
    evict_ctx = {s["context"] for s in S if s["cache_mode"] == "evict"}
    have = {(s["session_id"], s["policy"], s["context"], s["cache_mode"]) for s in S}
    for rows in (S, T):
        extra = []
        for x in rows:
            if x["policy"] in ("P1", "P2") and x["cache_mode"] == "retain" and x["context"] in evict_ctx \
                    and (x["session_id"], x["policy"], x["context"], "evict") not in have:
                extra.append({**x, "cache_mode": "evict", "reused_from_retain": True})
        rows += extra
    return S, T


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def session_table(S):
    rows = []
    for s in S:
        rows.append({"session_id": s["session_id"], "scenario": s["scenario"], "subtype": s["subtype"],
                     "policy": s["policy"], "context": s["context"], "cache": s["cache_mode"],
                     "step_pass": s["step_pass_rate"], "all_pass": s["all_pass"], "final_pass": s["final_pass"],
                     "latency_s": s["latency_s"], "latency_excl_verifier_s": s["latency_excl_verifier_s"],
                     "router_calls": s["router_calls"], "switches": s["switches"], "turns": s["n_turns"],
                     "A_calls": s["model_calls"]["A"], "B_calls": s["model_calls"]["B"]})
    return pd.DataFrame(rows)


def turn_table(T):
    rows = []
    for t in T:
        c = t["call"]
        sv = c["server"] or {}
        rr = t["router"] or {}
        rows.append({"session_id": t["session_id"], "policy": t["policy"], "context": t["context"],
                     "cache": t["cache_mode"], "turn": t["turn"], "step": t["step"], "attempt": t["attempt"],
                     "model": t["chosen"], "switch": t["switch"], "events": ",".join(t["events"]),
                     "router_called": bool(rr), "router_choice": rr.get("choice"), "router_raw": rr.get("raw"),
                     "router_s": rr.get("router_s", 0.0), "router_in_tok": rr.get("input_tokens", 0),
                     "router_parse_error": rr.get("parse_error", False),
                     "gate_applied": t["gate"].get("applied", False), "gate_s": t["gate"].get("gate_s", 0.0),
                     "gate_lookup_s": t["gate"].get("lookup_s", 0.0),
                     "gate_overrode": t["gate"].get("overrode_router", False),
                     "gate_saving_s": t["gate"].get("saving_s"),
                     "evict_s": t["evict_s"], "render_s": t["render_s"], "verify_s": t["verify_s"],
                     "ttft_s": c["ttft_s"], "call_s": c["total_s"], "prompt_tokens": c["prompt_tokens"],
                     "cached_tokens": c["cached_tokens"], "predicted_hit": t["predicted_hit"],
                     "completion_tokens": c["completion_tokens"], "finish": c["finish_reason"],
                     "srv_prefill_s": sv.get("vllm:request_prefill_time_seconds_sum"),
                     "srv_decode_s": sv.get("vllm:request_decode_time_seconds_sum"),
                     "code_found": t["code_found"],
                     "verifier": (t["verifier"] or {}).get("status"),
                     "turn_latency_s": t["turn_latency_s"]})
    df = pd.DataFrame(rows)
    df["recomputed_tokens"] = df["prompt_tokens"] - df["cached_tokens"].fillna(0)
    # client-side decode share = call - ttft (ttft includes queue + prefill + first token)
    df["decode_client_s"] = df["call_s"] - df["ttft_s"].fillna(df["call_s"])
    return df


def summarize(st, keys):
    g = st.groupby(keys)
    out = g.agg(n=("session_id", "count"), step_pass=("step_pass", "mean"), final_pass=("final_pass", "mean"),
                all_pass=("all_pass", "mean"), latency_s=("latency_s", "mean"),
                latency_excl_verifier_s=("latency_excl_verifier_s", "mean"),
                router_calls=("router_calls", "mean"), switches=("switches", "mean"),
                B_share=("B_calls", "sum"), turns=("turns", "mean"))
    out["B_share"] = g["B_calls"].sum() / g["turns"].sum()
    return out.reset_index()


def paired(st, cond_keys):
    out, cases = [], []
    for cond, sub in st.groupby(cond_keys):
        piv_q = sub.pivot_table(index="session_id", columns="policy", values="step_pass")
        piv_l = sub.pivot_table(index="session_id", columns="policy", values="latency_s")
        piv_f = sub.pivot_table(index="session_id", columns="policy", values="final_pass")
        for a, b in PAIRS:
            if a not in piv_q or b not in piv_q:
                continue
            m = piv_q[[a, b]].dropna().index
            dq = (piv_q.loc[m, a] - piv_q.loc[m, b]).values
            dl = (piv_l.loc[m, a] - piv_l.loc[m, b]).values
            df_ = (piv_f.loc[m, a].astype(float) - piv_f.loc[m, b].astype(float)).values
            rec = dict(zip(cond_keys, cond if isinstance(cond, tuple) else (cond,)))
            rec.update({"a": a, "b": b, "n": len(m), "d_step_pass": dq.mean(), "d_step_pass_ci": boot(dq),
                        "d_final_pass": df_.mean(), "d_latency_s": dl.mean(), "d_latency_ci": boot(dl),
                        "rel_latency": piv_l.loc[m, a].sum() / piv_l.loc[m, b].sum(),
                        "n_better": int((dq > 0).sum()), "n_worse": int((dq < 0).sum()), "n_same": int((dq == 0).sum())})
            out.append(rec)
            for sid in m:
                d = piv_q.loc[sid, a] - piv_q.loc[sid, b]
                if d != 0:
                    cases.append({**{k: rec[k] for k in cond_keys}, "a": a, "b": b, "session_id": sid,
                                  "d_step_pass": d, "d_latency_s": piv_l.loc[sid, a] - piv_l.loc[sid, b]})
    return pd.DataFrame(out), pd.DataFrame(cases)


def change_lag(S, T, sessions_meta):
    """Turns from the first step whose difficulty becomes H (after an E) until B is first chosen."""
    by = collections.defaultdict(list)
    for t in T:
        by[(t["session_id"], t["policy"], t["context"], t["cache_mode"])].append(t)
    rows = []
    for (sid, p, ctx, cm), ts in by.items():
        meta = sessions_meta[sid]
        pat = meta["pattern"]
        for k in range(1, len(pat)):
            if pat[k - 1] == "E" and pat[k] == "H":
                exposed = meta["structure"][k] == "N"
                after = sorted([t for t in ts if t["step"] >= k], key=lambda t: t["turn"])
                lag = next((i for i, t in enumerate(after) if t["chosen"] == "B"), None)
                rows.append({"session_id": sid, "scenario": meta["scenario"], "policy": p, "context": ctx,
                             "cache": cm, "change_step": k, "change_exposed": exposed,
                             "turns_to_B": lag, "router_called_at_change": bool(after and after[0]["router"])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def fig_quality_latency(summ, out, title):
    scen = sorted(summ["scenario"].unique())
    fig, axs = plt.subplots(1, len(scen), figsize=(4 * len(scen), 3.6), sharey=True)
    for ax, sc in zip(np.atleast_1d(axs), scen):
        d = summ[summ["scenario"] == sc]
        for _, r in d.iterrows():
            ax.scatter(r["latency_s"], r["step_pass"], color=COL[r["policy"]], s=40, zorder=3)
            ax.annotate(r["policy"], (r["latency_s"], r["step_pass"]), fontsize=8, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_title(sc, fontsize=10)
        ax.set_xlabel("mean session latency (s)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axs)[0].set_ylabel("mean step pass rate")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_breakdown(tt, out, title):
    comp = collections.OrderedDict()
    for p in POL:
        d = tt[tt["policy"] == p]
        if d.empty:
            continue
        ns = d["session_id"].nunique()
        comp[p] = {"router": d["router_s"].sum() / ns,
                   "A prefill(TTFT)": d.loc[d.model == "A", "ttft_s"].sum() / ns,
                   "A decode": d.loc[d.model == "A", "decode_client_s"].sum() / ns,
                   "B prefill(TTFT)": d.loc[d.model == "B", "ttft_s"].sum() / ns,
                   "B decode": d.loc[d.model == "B", "decode_client_s"].sum() / ns,
                   "verifier": d["verify_s"].sum() / ns,
                   "gate+render+evict": (d["gate_s"] + d["render_s"] + d["evict_s"]).sum() / ns}
    df = pd.DataFrame(comp).T
    ax = df.plot(kind="bar", stacked=True, figsize=(8, 4),
                 color=["#d62728", "#aec7e8", "#1f77b4", "#c7c7c7", "#000000", "#2ca02c", "#ff7f0e"])
    ax.set_ylabel("mean seconds per session")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    return df


def fig_rank(summ_cond, out):
    """Policy order by latency and quality across context x cache conditions."""
    conds = [c for c in [("short", "retain"), ("short", "evict"), ("long", "retain"), ("long", "evict")]
             if ((summ_cond["context"] == c[0]) & (summ_cond["cache"] == c[1])).any()]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric in zip(axs, ["latency_s", "step_pass"]):
        for p in POL:
            ys = []
            for c in conds:
                d = summ_cond[(summ_cond["context"] == c[0]) & (summ_cond["cache"] == c[1])]
                ys.append(d.set_index("policy")[metric].get(p, np.nan))
            if not np.all(np.isnan(ys)):
                ax.plot(range(len(conds)), ys, "-o", color=COL[p], label=p)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([f"{a}\n{b}" for a, b in conds])
        ax.set_title({"latency_s": "mean session latency (s)", "step_pass": "mean step pass rate"}[metric], fontsize=10)
        ax.grid(alpha=0.3)
    axs[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_timeline(T, sid, ctx, cm, out, steps_meta):
    ts = [t for t in T if t["session_id"] == sid and t["context"] == ctx and t["cache_mode"] == cm]
    pols = [p for p in POL if any(t["policy"] == p for t in ts)]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(pols) + 1.4))
    for i, p in enumerate(pols):
        for t in sorted([t for t in ts if t["policy"] == p], key=lambda t: t["turn"]):
            x = t["turn"]
            ax.barh(i, 0.9, left=x, color="#9ecae1" if t["chosen"] == "A" else "#08306b", edgecolor="white")
            if t["router"]:
                ax.plot(x + 0.15, i + 0.3, "v", color="#d62728", ms=5)
            if t["verifier"] and t["verifier"]["status"] != "PASS":
                ax.plot(x + 0.75, i - 0.25, "x", color="#ff7f0e", ms=6, mew=2)
            if "new_task" in t["events"]:
                ax.plot(x + 0.05, i - 0.3, "^", color="#2ca02c", ms=5)
            ax.text(x + 0.45, i, f"s{t['step']}" + ("r" if t["attempt"] else ""), ha="center", va="center",
                    fontsize=7, color="white" if t["chosen"] == "B" else "black")
    ax.set_yticks(range(len(pols)))
    ax.set_yticklabels(pols)
    ax.invert_yaxis()
    ax.set_xlabel("turn (light = A, dark = B; red ▼ router call; green ▲ new task; orange × verifier fail)")
    ax.set_title(f"{sid}  pattern={steps_meta['pattern']} structure={steps_meta['structure']}  ({ctx}/{cm})",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions-file", default="data/eval/eval.jsonl")
    ap.add_argument("--timeline", default="")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    S, T = load(a.runs.split(","))
    meta = {json.loads(l)["session_id"]: json.loads(l) for l in open(ROOT / a.sessions_file)}
    st, tt = session_table(S), turn_table(T)
    st.to_csv(out / "sessions.csv", index=False)
    tt.to_csv(out / "turns.csv", index=False)
    s_sc = summarize(st, ["context", "cache", "scenario", "policy"])
    s_all = summarize(st, ["context", "cache", "policy"])
    s_sc.to_csv(out / "summary_by_scenario.csv", index=False)
    s_all.to_csv(out / "summary_overall.csv", index=False)
    pr, cases = paired(st, ["context", "cache", "scenario"])
    pr.to_csv(out / "paired_by_scenario.csv", index=False)
    cases.to_csv(out / "paired_cases.csv", index=False)
    pa, _ = paired(st, ["context", "cache"])
    pa.to_csv(out / "paired_overall.csv", index=False)
    lag = change_lag(S, T, meta)
    lag.to_csv(out / "change_lag.csv", index=False)
    # integrity
    integ = {"turns": len(tt), "sessions": len(st),
             "hit_pred_mismatch": int((tt["predicted_hit"] != tt["cached_tokens"]).sum()),
             "hit_null": int(tt["cached_tokens"].isna().sum()),
             "truncated": int((tt["finish"] == "length").sum()), "no_code": int((~tt["code_found"]).sum()),
             "router_parse_errors": int(tt["router_parse_error"].sum()),
             "gate_overrides": int(tt["gate_overrode"].sum()), "gate_applied": int(tt["gate_applied"].sum())}
    (out / "integrity.json").write_text(json.dumps(integ, indent=1))
    for (ctx, cm), d in s_sc.groupby(["context", "cache"]):
        fig_quality_latency(d, out / f"fig1_quality_latency_{ctx}_{cm}.png", f"{ctx} context / {cm}")
        fig_breakdown(tt[(tt.context == ctx) & (tt.cache == cm)], out / f"fig2_breakdown_{ctx}_{cm}.png",
                      f"cost breakdown per session, {ctx}/{cm}").to_csv(out / f"breakdown_{ctx}_{cm}.csv")
    fig_rank(s_all, out / "fig3_rank_by_condition.png")
    for spec in filter(None, a.timeline.split(";")):
        sid, ctx, cm = spec.split(",")
        fig_timeline(T, sid, ctx, cm, out / f"fig4_timeline_{sid}_{ctx}_{cm}.png", meta[sid])
    print(json.dumps(integ))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(s_all.round(3).to_string())


if __name__ == "__main__":
    main()
