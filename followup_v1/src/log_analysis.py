"""Part 4: analysis of existing logs only (no new inference).

    python log_analysis.py <run>[,<run>...] <out_dir>

Definitions (kept separate on purpose):
- initial failure: the policy's final step-0 submission fails the hidden evaluator.
- initial selection error (diagnostic proxy): the model used for step 0 attempt 0 is the one that did
  worse on that same first request, judged by the P1 (A) and P2 (B) runs of the same session (step 0
  input is identical across policies). "A chosen, A fails, B passes" or "B chosen, B fails, A passes".
  Undefined when A and B have the same outcome.
- re-routing: a router call after turn 0. Co-occurrence of re-routing and later recovery is reported as
  co-occurrence, not as an effect.
"""
import collections
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SESS = {json.loads(l)["session_id"]: json.loads(l) for l in open(ROOT / "data/eval/eval.jsonl")}


def load(runs):
    S, T = [], []
    for r in runs:
        base = ROOT / "results" / r if (ROOT / "results" / r).exists() else ROOT / "followup_v1/results" / r
        S += [json.loads(l) for l in open(base / "sessions.jsonl")]
        T += [json.loads(l) for l in open(base / "turns.jsonl")]
    return S, T


def attrs(sid):
    s = SESS[sid]
    hidden = any(s["pattern"][k] != s["pattern"][k - 1] and s["structure"][k] == "F" for k in range(1, 5))
    exposed = any(s["pattern"][k] != s["pattern"][k - 1] and s["structure"][k] == "N" for k in range(1, 5))
    return {"scenario": s["scenario"], "verifier": s["verifier"], "hidden_change": hidden, "exposed_change": exposed}


def main(runs, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    S, T = load(runs)
    key = lambda x: (x["session_id"], x["policy"], x["context"], x["cache_mode"])
    by_s = {key(s): s for s in S}
    by_t = collections.defaultdict(list)
    for t in T:
        by_t[key(t)].append(t)
    for v in by_t.values():
        v.sort(key=lambda t: t["turn"])
    ref = {}  # (sid, ctx) -> P1/P2 step-0 outcome (short/retain and long/retain refs)
    for (sid, p, ctx, cm), s in by_s.items():
        if p in ("P1", "P2") and cm == "retain":
            ref[(sid, ctx, p)] = s["steps"][0]["pass"]

    # ---- 4.1 first-step failure -> first re-routing
    rows = []
    for k, s in by_s.items():
        sid, p, ctx, cm = k
        if p in ("P1", "P2"):
            continue
        ts = by_t[k]
        m0 = ts[0]["chosen"]
        a_ok, b_ok = ref.get((sid, ctx, "P1")), ref.get((sid, ctx, "P2"))
        sel_err = None
        if a_ok is not None and b_ok is not None and a_ok != b_ok:
            sel_err = (m0 == "A" and not a_ok) or (m0 == "B" and not b_ok)
        if s["steps"][0]["pass"]:
            continue
        after = [t for t in ts if t["turn"] > 0 and t["router"]]
        first = after[0] if after else None
        later = [e["pass"] for e in s["steps"][1:]]
        rows.append({"session_id": sid, "policy": p, "context": ctx, "cache": cm, **attrs(sid),
                     "model_step0": m0, "initial_selection_error": sel_err,
                     "first_reroute_turn": first["turn"] if first else None,
                     "first_reroute_step": first["step"] if first else None,
                     "first_reroute_reason": (first.get("route_reason") or ",".join(first["events"]) or "schedule")
                     if first else None,
                     "reroute_changed_model": (first["chosen"] != first["active_before"]) if first else None,
                     "never_rerouted": first is None,
                     "later_step_pass": sum(later) / len(later), "final_pass": s["final_pass"]})
    f = pd.DataFrame(rows)
    f.to_csv(out / "first_failure_reroute.csv", index=False)
    g = f.groupby(["context", "cache", "policy"]).agg(
        n_initial_fail=("session_id", "count"),
        initial_sel_error=("initial_selection_error", lambda x: int((x == True).sum())),  # noqa: E712
        sel_error_undefined=("initial_selection_error", lambda x: int(x.isna().sum())),
        never_rerouted=("never_rerouted", "mean"),
        median_first_reroute_turn=("first_reroute_turn", "median"),
        reroute_changed_model=("reroute_changed_model", lambda x: x.dropna().mean()),
        later_step_pass=("later_step_pass", "mean"), final_pass=("final_pass", "mean")).reset_index()
    g.to_csv(out / "first_failure_summary.csv", index=False)
    # co-occurrence table: changed vs not changed at first reroute
    co = f[f.first_reroute_turn.notna()].groupby(["context", "cache", "policy", "reroute_changed_model"]).agg(
        n=("session_id", "count"), later_step_pass=("later_step_pass", "mean"), final_pass=("final_pass", "mean")
    ).reset_index()
    co.to_csv(out / "first_failure_cooccurrence.csv", index=False)

    # ---- 4.2 router calls that kept the model; router vs model time share
    r2 = []
    for k, ts in by_t.items():
        sid, p, ctx, cm = k
        if p in ("P1", "P2"):
            continue
        calls = [t for t in ts if t["router"]]
        later = [t for t in calls if t["turn"] > 0]
        r2.append({"policy": p, "context": ctx, "cache": cm, "router_calls": len(calls),
                   "later_calls": len(later), "later_kept": sum(t["chosen"] == t["active_before"] for t in later),
                   "router_s": sum(t["router"]["router_s"] for t in calls),
                   "model_s": sum(t["call"]["total_s"] for t in ts),
                   "session_s": sum(t["turn_latency_s"] for t in ts)})
    r2 = pd.DataFrame(r2).groupby(["context", "cache", "policy"]).sum().reset_index()
    r2["kept_rate_after_turn0"] = r2.later_kept / r2.later_calls
    r2["router_share_of_session"] = r2.router_s / r2.session_s
    r2["router_s_per_call"] = r2.router_s / r2.router_calls
    r2.to_csv(out / "router_calls_and_time.csv", index=False)

    # ---- 4.3 missed opportunities: hard step with no router call before it in the step, by event policy
    miss = []
    for k, ts in by_t.items():
        sid, p, ctx, cm = k
        if p not in ("P5", "P6", "P8", "P9", "P10", "P3"):
            continue
        for st in SESS[sid]["steps"]:
            t0 = next(t for t in ts if t["step"] == st["step"] and t["attempt"] == 0)
            if st["difficulty"] == "hard" and t0["chosen"] == "A" and not t0["router"]:
                miss.append({"session_id": sid, "policy": p, "context": ctx, "cache": cm, "step": st["step"],
                             "kind": st["kind"], **attrs(sid),
                             "step_pass": by_s[k]["steps"][st["step"]]["pass"]})
    m = pd.DataFrame(miss)
    m.to_csv(out / "hard_step_on_A_without_router_call.csv", index=False)
    if len(m):
        m.groupby(["context", "cache", "policy", "kind"]).agg(n=("step", "count"), step_pass=("step_pass", "mean")) \
            .reset_index().to_csv(out / "hard_step_on_A_without_router_call_summary.csv", index=False)

    # ---- 4.4 event-based vs every-turn and vs always-A, by verifier and boundary exposure
    rows = []
    for (sid, p, ctx, cm), s in by_s.items():
        if p != "P6":
            continue
        for b in ("P4", "P1", "P2"):
            kb = (sid, b, ctx, cm if b not in ("P1", "P2") else "retain")
            if kb not in by_s:
                continue
            o = by_s[kb]
            rows.append({"session_id": sid, "vs": b, "context": ctx, "cache": cm, **attrs(sid),
                         "d_step_pass": s["step_pass_rate"] - o["step_pass_rate"],
                         "d_latency_s": s["latency_s"] - o["latency_s"]})
    e = pd.DataFrame(rows)
    e.to_csv(out / "event_vs_others_by_attrs.csv", index=False)
    if len(e):
        e.groupby(["context", "cache", "vs", "verifier", "hidden_change"]).agg(
            n=("session_id", "count"), d_step_pass=("d_step_pass", "mean"), d_latency_s=("d_latency_s", "mean"),
            n_better=("d_step_pass", lambda x: int((x > 0).sum())), n_worse=("d_step_pass", lambda x: int((x < 0).sum()))
        ).reset_index().to_csv(out / "event_vs_others_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(g.round(3).to_string())
        print(co.round(3).to_string())
        print(r2.round(3).to_string())
        if len(m):
            print(pd.read_csv(out / "hard_step_on_A_without_router_call_summary.csv").round(3).to_string())
        if len(e):
            print(pd.read_csv(out / "event_vs_others_summary.csv").round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2])
