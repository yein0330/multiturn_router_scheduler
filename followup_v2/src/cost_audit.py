"""followup_v2 Part 2: cost audit of existing logs (no new inference).

    python cost_audit.py <out_dir>

Call classes, in actual call order within a session (retries included):
  first_call    session's first model call
  first_switch  switch to a model not used earlier in the session
  return        switch back to a model used earlier in the session
  stay          same model as the previous call
TTFT = client time to first streamed text token (includes queueing, prefill and the first decode step;
it is NOT pure prefill). gen = call total - TTFT.
Session latency (existing definition) = sum over turns of router + gate + evict + render + call + verify,
so the per-session decomposition below is exact up to float rounding (residual reported).
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ["evalA", "evalB_long_retain", "evalB_short_evict", "evalB_long_evict"]
SESS = {json.loads(l)["session_id"]: json.loads(l) for l in open(ROOT / "data/eval/eval.jsonl")}


def load():
    S, T = [], []
    for r in RUNS:
        S += [json.loads(l) for l in open(ROOT / "results" / r / "sessions.jsonl")]
        T += [json.loads(l) for l in open(ROOT / "results" / r / "turns.jsonl")]
    return S, T


def classify(T):
    by = collections.defaultdict(list)
    for t in T:
        by[(t["session_id"], t["policy"], t["context"], t["cache_mode"])].append(t)
    rows = []
    for k, ts in by.items():
        ts.sort(key=lambda t: t["turn"])
        used, prev = set(), None
        for t in ts:
            m = t["chosen"]
            if prev is None:
                cls = "first_call"
            elif m != prev:
                cls = "return" if m in used else "first_switch"
            else:
                cls = "stay"
            c = t["call"]
            ttft = c["ttft_s"] if c["ttft_s"] is not None else c["total_s"]
            rows.append({"session_id": k[0], "policy": k[1], "context": k[2], "cache": k[3], "turn": t["turn"],
                         "step": t["step"], "attempt": t["attempt"], "model": m, "cls": cls,
                         "ttft_s": ttft, "gen_s": c["total_s"] - ttft, "call_s": c["total_s"],
                         "prompt_tokens": c["prompt_tokens"], "completion_tokens": c["completion_tokens"],
                         "hit_tokens": c["cached_tokens"] or 0,
                         "miss_tokens": c["prompt_tokens"] - (c["cached_tokens"] or 0),
                         "router_s": t["router"]["router_s"] if t["router"] else 0.0,
                         "verify_s": t["verify_s"], "gate_s": t["gate"].get("gate_s", 0.0),
                         "render_s": t["render_s"], "evict_s": t["evict_s"],
                         "is_retry": t["attempt"] > 0, "turn_latency_s": t["turn_latency_s"]})
            used.add(m)
            prev = m
    return pd.DataFrame(rows)


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    S, T = load()
    df = classify(T)
    df.to_csv(out / "calls_classified.csv", index=False)
    g = df.groupby(["context", "cache", "policy", "cls", "model"]).agg(
        calls=("turn", "count"), sessions=("session_id", "nunique"), retries=("is_retry", "sum"),
        ttft_sum_s=("ttft_s", "sum"), ttft_mean_s=("ttft_s", "mean"), gen_sum_s=("gen_s", "sum"),
        prompt_tokens=("prompt_tokens", "sum"), completion_tokens=("completion_tokens", "sum"),
        hit_tokens=("hit_tokens", "sum"), miss_tokens=("miss_tokens", "sum"),
        router_s=("router_s", "sum"), verify_s=("verify_s", "sum"),
        other_s=("gate_s", "sum")).reset_index()
    g["other_s"] = df.groupby(["context", "cache", "policy", "cls", "model"]).apply(
        lambda x: (x.gate_s + x.render_s + x.evict_s).sum()).values
    g.to_csv(out / "call_class_table.csv", index=False)

    # ---- per-session P4 - P2 decomposition (P2 retain reused for evict: it never switches)
    comp = ["router_s", "gate_s", "render_s", "evict_s", "verify_s"]
    sess_lat = {(s["session_id"], s["policy"], s["context"], s["cache_mode"]): s["latency_s"] for s in S}
    rows = []
    for (ctx, cm), sub in df.groupby(["context", "cache"]):
        for sid in SESS:
            a = sub[(sub.session_id == sid) & (sub.policy == "P4")]
            b = df[(df.session_id == sid) & (df.policy == "P2") & (df.context == ctx) & (df.cache == "retain")]
            if a.empty or b.empty:
                continue
            r = {"context": ctx, "cache": cm, "session_id": sid, "scenario": SESS[sid]["scenario"]}
            for c in comp:
                r["d_" + c] = a[c].sum() - b[c].sum()
            for m in "AB":
                for part in ("ttft_s", "gen_s"):
                    r[f"d_{m}_{part}"] = a.loc[a.model == m, part].sum() - b.loc[b.model == m, part].sum()
            for cls in ("first_call", "first_switch", "return", "stay"):
                r[f"P4_B_ttft_{cls}"] = a.loc[(a.model == "B") & (a.cls == cls), "ttft_s"].sum()
            r["P2_B_ttft_first_call"] = b.loc[b.cls == "first_call", "ttft_s"].sum()
            r["P2_B_ttft_stay"] = b.loc[b.cls == "stay", "ttft_s"].sum()
            r["d_turns"] = len(a) - len(b)
            r["d_completion_tokens"] = a.completion_tokens.sum() - b.completion_tokens.sum()
            r["d_B_completion_tokens"] = a.loc[a.model == "B", "completion_tokens"].sum() - b.completion_tokens.sum()
            r["P4_A_completion_tokens"] = a.loc[a.model == "A", "completion_tokens"].sum()
            r["d_miss_tokens"] = a.miss_tokens.sum() - b.miss_tokens.sum()
            parts = [k for k in r if k.startswith("d_") and k.endswith("_s")]
            r["sum_parts_s"] = sum(r[k] for k in parts)
            r["d_latency_s"] = sess_lat[(sid, "P4", ctx, cm)] - sess_lat[(sid, "P2", ctx, "retain")]
            r["residual_s"] = r["d_latency_s"] - r["sum_parts_s"]
            rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(out / "paired_P4_minus_P2_by_session.csv", index=False)
    cols = [c for c in d.columns if c.startswith(("d_", "P4_", "P2_", "sum_", "residual"))]
    m = d.groupby(["context", "cache"])[cols].mean().reset_index()
    m.to_csv(out / "paired_P4_minus_P2_mean.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(g[(g.policy.isin(["P2", "P4", "P6"]))].round(3).to_string())
        print(m.round(3).T.to_string())
        print("max |residual| s:", float(d.residual_s.abs().max()))


if __name__ == "__main__":
    main(sys.argv[1])
