"""Part 1: audit + tables for stage A and the B2 rerun (overall and per scenario).

    python b2_tables.py <out_dir> [run ...]     (default runs: evalA evalB_long_retain evalB_short_evict evalB_long_evict)

Latency definitions:
- latency_s (primary; = original definition): sum over turns of router + gate + evict + render + LLM call +
  verifier. The per-run isolation reset happens before the session clock starts, so it was never included.
- latency_excl_evict_s: latency_s minus mid-session eviction resets (evict condition only; diagnostic).
- wall_incl_instrumentation_s: includes metric scrapes (kept for reference).
First-attempt step success re-scores the attempt-0 code of each step with the hidden evaluator
(differs from final only in S5, where one verifier retry exists).
A run is "complete" for a condition only if all 50 sessions x its policies are present exactly once.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mrp import sandbox  # noqa: E402

SESS = {json.loads(l)["session_id"]: json.loads(l) for l in open(ROOT / "data/eval/eval.jsonl")}
EXPECTED = {"evalA": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
            "evalB_long_retain": ["P1", "P2", "P4", "P6", "P7", "P8"],
            "evalB_short_evict": ["P4", "P6", "P7", "P8"], "evalB_long_evict": ["P4", "P6", "P7", "P8"]}


def audit(run):
    d = ROOT / "results" / run
    if not (d / "sessions.jsonl").exists():
        return {"run": run, "present": False}, [], []
    S = [json.loads(l) for l in open(d / "sessions.jsonl")]
    T = [json.loads(l) for l in open(d / "turns.jsonl")]
    c = collections.Counter((s["session_id"], s["policy"]) for s in S)
    exp = {(sid, p) for sid in SESS for p in EXPECTED[run]}
    a = {"run": run, "present": True, "session_rows": len(S), "unique_runs": len(c),
         "expected_runs": len(exp), "missing_runs": len(exp - set(c)),
         "duplicate_runs": sum(v > 1 for v in c.values()), "turns": len(T),
         "hit_mismatch": sum(t["predicted_hit"] != t["call"]["cached_tokens"] for t in T),
         "hit_null": sum(t["call"]["cached_tokens"] is None for t in T),
         "truncated": sum(t["call"]["finish_reason"] == "length" for t in T),
         "http_errors": sum(bool(t["call"]["error"]) for t in T),
         "no_code": sum(not t["code_found"] for t in T),
         "router_parse_errors": sum(bool(t["router"] and t["router"]["parse_error"]) for t in T)}
    a["complete"] = a["missing_runs"] == 0 and a["duplicate_runs"] == 0
    a["clean"] = a["hit_mismatch"] == 0 and a["http_errors"] == 0
    return a, S, T


def first_attempt_pass(T):
    out = {}
    for t in T:
        if t["attempt"] == 0:
            s = SESS[t["session_id"]]
            st = s["steps"][t["step"]]
            if not s["verifier"]:
                continue  # first attempt == final outside S5
            code = sandbox.extract_code(t["assistant_text"])
            out[(t["session_id"], t["policy"], t["context"], t["cache_mode"], t["step"])] = \
                sandbox.evaluate(code, s["hidden_tables"], st["gold_hidden"])["pass"]
    return out


def tables(S, T, fa):
    tk = lambda t: (t["session_id"], t["policy"], t["context"], t["cache_mode"])
    by_t = collections.defaultdict(list)
    for t in T:
        by_t[tk(t)].append(t)
    rows = []
    for s in S:
        k = (s["session_id"], s["policy"], s["context"], s["cache_mode"])
        ts = by_t[k]
        fa_steps = [fa.get(k + (e["step"],), e["pass"]) for e in s["steps"]]
        ba = [t for t in ts if t["router"] and t["active_before"] == "B" and t["router"]["choice"] == "A"]
        rows.append({"session_id": s["session_id"], "scenario": s["scenario"], "policy": s["policy"],
                     "context": s["context"], "cache": s["cache_mode"],
                     "first_attempt_step_pass": np.mean(fa_steps), "after_retry_step_pass": s["step_pass_rate"],
                     "final_step_pass": float(s["final_pass"]), "all_steps_pass": float(s["all_pass"]),
                     "latency_s": s["latency_s"],
                     "latency_excl_evict_s": s["latency_s"] - sum(t["evict_s"] for t in ts),
                     "wall_incl_instrumentation_s": s["wall_incl_instrumentation_s"],
                     "router_calls": s["router_calls"], "router_s": sum(t["router"]["router_s"] for t in ts if t["router"]),
                     "switches": s["switches"],
                     "recomputed_tokens": sum(t["call"]["prompt_tokens"] - (t["call"]["cached_tokens"] or 0) for t in ts),
                     "B_to_A_recs": len(ba),
                     "gate_allowed": sum(t["gate"].get("applied") and not t["gate"].get("overrode_router") for t in ba),
                     "gate_held": sum(bool(t["gate"].get("overrode_router")) for t in ba),
                     "fallbacks": sum(bool(t["router"] and t["router"]["parse_error"]) for t in ts)})
    return pd.DataFrame(rows)


def agg(df, keys):
    g = df.groupby(keys)
    out = g.agg(n_sessions=("session_id", "count"),
                first_attempt_step_pass=("first_attempt_step_pass", "mean"),
                after_retry_step_pass=("after_retry_step_pass", "mean"),
                final_step_pass=("final_step_pass", "mean"), all_steps_pass=("all_steps_pass", "mean"),
                latency_mean_s=("latency_s", "mean"), latency_median_s=("latency_s", "median"),
                latency_excl_evict_mean_s=("latency_excl_evict_s", "mean"),
                wall_incl_instr_mean_s=("wall_incl_instrumentation_s", "mean"),
                router_calls=("router_calls", "mean"), router_s=("router_s", "mean"),
                switches=("switches", "mean"), recomputed_tokens=("recomputed_tokens", "mean"),
                B_to_A_recs=("B_to_A_recs", "sum"), gate_allowed=("gate_allowed", "sum"),
                gate_held=("gate_held", "sum"), fallbacks=("fallbacks", "sum"))
    return out.reset_index()


def paired(df, a, b, keys):
    """Same session, same condition. b may be P1/P2 retain reused for evict (marked)."""
    rows = []
    for cond, sub in df.groupby(keys):
        pa = sub[sub.policy == a].set_index("session_id")
        pb = sub[sub.policy == b].set_index("session_id")
        m = pa.index.intersection(pb.index)
        if len(m) == 0:
            continue
        dq = pa.loc[m, "after_retry_step_pass"] - pb.loc[m, "after_retry_step_pass"]
        dl = pa.loc[m, "latency_s"] - pb.loc[m, "latency_s"]
        rows.append({**dict(zip(keys, cond if isinstance(cond, tuple) else (cond,))), "a": a, "b": b, "n": len(m),
                     "d_step_pass": dq.mean(), "d_latency_s": dl.mean(),
                     "n_better": int((dq > 0).sum()), "n_worse": int((dq < 0).sum())})
    return rows


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    runs = sys.argv[2:] or list(EXPECTED)
    audits, S, T = [], [], []
    for r in runs:
        a, s, t = audit(r)
        audits.append(a)
        if a.get("present") and a["complete"] and a["clean"]:
            S += s; T += t
        elif a.get("present"):
            a["note"] = "EXCLUDED from tables: incomplete or not clean"
    pd.DataFrame(audits).to_csv(out / "audit.csv", index=False)
    print(pd.DataFrame(audits).to_string())
    if not S:
        return
    df = tables(S, T, first_attempt_pass(T))
    # P1/P2 evict cells reuse retain runs of the same context (never switch) - flagged
    reuse = []
    for ctx in df[df.cache == "evict"].context.unique():
        r = df[(df.context == ctx) & (df.cache == "retain") & df.policy.isin(["P1", "P2"])].copy()
        r["cache"] = "evict"; r["reused_from_retain"] = True
        reuse.append(r)
    if reuse:
        df = pd.concat([df] + reuse, ignore_index=True)
    df.to_csv(out / "session_rows.csv", index=False)
    o = agg(df, ["context", "cache", "policy"])
    sc = agg(df, ["context", "cache", "scenario", "policy"])
    o.to_csv(out / "table_overall.csv", index=False)
    sc.to_csv(out / "table_by_scenario.csv", index=False)
    pr = []
    for a, b in [("P4", "P2"), ("P4", "P1"), ("P6", "P4"), ("P7", "P4"), ("P8", "P6"), ("P8", "P7"), ("P6", "P2")]:
        pr += paired(df, a, b, ["context", "cache"])
    pd.DataFrame(pr).to_csv(out / "paired_within_condition.csv", index=False)
    # cross-condition: same session & policy across conditions (only complete conditions are loaded)
    piv = df.pivot_table(index=["session_id", "policy"], columns=["context", "cache"],
                         values=["after_retry_step_pass", "latency_s"])
    piv.to_csv(out / "same_session_across_conditions.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(o.round(3).to_string())
        print(pd.DataFrame(pr).round(3).to_string())


if __name__ == "__main__":
    main()
