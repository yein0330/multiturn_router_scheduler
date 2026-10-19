"""Stage A analysis (CPU). -> followup_v4/results/stageA_analysis/
Format errors = tool errors from malformed arguments (missing/unknown field, unknown op, bad arguments,
type errors) + turns without a parsable tool call. Semantic errors (e.g., a misspelled real column) are
counted under tool errors but not under format errors.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V4 = Path(__file__).resolve().parents[1]
ROOT = V4.parent
OUT = V4 / "results/stageA_analysis"
FMT = re.compile(r"unknown column 'None|unknown op|bad arguments|must be a list|must be an object|must be non-negative|"
                 r"needs a numeric column and numeric value|arguments must be an object|invalid tool call JSON")


def load():
    S = [json.loads(l) for l in open(V4 / "results/stageA/sessions.jsonl")]
    C = [json.loads(l) for l in open(V4 / "results/stageA/calls.jsonl")]
    return S, C


def per_run(S, C):
    by = collections.defaultdict(list)
    for c in C:
        by[(c["session_id"], c["context"], c["schema"], c["policy"])].append(c)
    rows = []
    for s in S:
        k = (s["session_id"], s["context"], s["schema"], s["policy"])
        cs = by[k]
        tool = [t for c in cs for t in c["tools"]]
        terr = [t for t in tool if t["error"]]
        ferr_req, rep, comp = collections.defaultdict(int), 0, collections.Counter()
        seen = collections.Counter()
        for c in cs:
            f = sum(1 for t in c["tools"] if t["error"] and FMT.search(t["error"])) + (c["n_tool_calls"] == 0)
            ferr_req[c["request"]] += f
            for t in c["tools"]:
                if t["error"]:
                    key = (c["request"], re.sub(r"'[^']*'", "'X'", t["error"]))
                    seen[key] += 1
        rep = sum(v - 1 for v in seen.values() if v > 1)
        r = {"session_id": s["session_id"], "context": s["context"], "schema": s["schema"], "policy": s["policy"],
             "q": s["q"], "S": float(s["S"]), "latency_s": s["latency_s"], "wall_s": s["wall_s"], "n_calls": s["n_calls"],
             "router_calls": s["router_calls"], "router_s": s["router_s"], "switches": s["switches"],
             "A_calls": s["model_calls"]["A"], "B_calls": s["model_calls"]["B"],
             "A_out_tokens": sum(c["completion_tokens"] for c in cs if c["chosen"] == "A"),
             "B_out_tokens": sum(c["completion_tokens"] for c in cs if c["chosen"] == "B"),
             "model_s": sum(c["total_s"] for c in cs), "tool_s": sum(c["tool_exec_s"] for c in cs),
             "hit_tokens": sum(c["cached_tokens"] or 0 for c in cs), "prompt_tokens": sum(c["prompt_tokens"] for c in cs),
             "tool_calls": len(tool), "tool_errors": len(terr),
             "tool_error_rate": len(terr) / len(tool) if tool else 0.0,
             "format_errors": sum(ferr_req.values()),
             "requests_with_format_error": sum(1 for v in ferr_req.values() if v > 0) / len(s["requests"]),
             "repeated_same_error": rep,
             "budget_exhausted_rate": sum(not r["submitted"] for r in s["requests"]) / len(s["requests"]),
             "cache_mismatch": sum(c["pred_hit"][c["chosen"]] != c["cached_tokens"] for c in cs),
             "truncated": sum(c["finish"] == "length" for c in cs), "http_errors": sum(bool(c["error"]) for c in cs),
             "router_fallback": sum(bool(c["router"] and c["router"]["parse_error"]) for c in cs)}
        rows.append(r)
    return pd.DataFrame(rows)


def boot(x, n=2000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(0)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S, C = load()
    d = per_run(S, C)
    d.to_csv(OUT / "runs.csv", index=False)
    cols = ["q", "S", "latency_s", "wall_s", "n_calls", "router_calls", "router_s", "switches", "A_calls", "B_calls",
            "A_out_tokens", "B_out_tokens", "tool_error_rate", "format_errors", "requests_with_format_error",
            "repeated_same_error", "budget_exhausted_rate"]
    summ = d.groupby(["context", "schema", "policy"])[cols].mean().reset_index()
    summ["n"] = d.groupby(["context", "schema", "policy"]).size().values
    summ.to_csv(OUT / "summary.csv", index=False)
    pairs = [("T1", "T0", p) for p in ("always_A", "always_B", "follow_R0", "follow_R1")]
    out = []
    for ctx in ("short", "long"):
        dd = d[d.context == ctx]
        piv = {m: dd.pivot_table(index="session_id", columns=["schema", "policy"], values=m)
               for m in ("q", "S", "latency_s", "format_errors", "budget_exhausted_rate")}
        comps = [((a, p), (b, p)) for a, b, p in pairs] + \
                [((sch, "follow_R1"), (sch, "follow_R0")) for sch in ("T0", "T1")] + \
                [((sch, "follow_R1"), (sch, "always_B")) for sch in ("T0", "T1")] + \
                [((sch, "follow_R0"), (sch, "always_B")) for sch in ("T0", "T1")] + \
                [((sch, "always_B"), (sch, "always_A")) for sch in ("T0", "T1")] + [(("T1", "follow_R1"), ("T0", "follow_R0"))]
        for a, b in comps:
            rec = {"context": ctx, "a": "/".join(a), "b": "/".join(b)}
            for m, P in piv.items():
                if a in P and b in P:
                    x = (P[a] - P[b]).dropna()
                    rec[f"d_{m}"] = float(x.mean())
                    rec[f"d_{m}_ci"] = boot(x)
                    rec["n"] = len(x)
            out.append(rec)
    pr = pd.DataFrame(out)
    pr.to_csv(OUT / "paired.csv", index=False)
    audit = {"runs": len(d), "expected": 160, "unique": int(d[["session_id", "context", "schema", "policy"]].drop_duplicates().shape[0]),
             "calls": len(C), "router_calls": int(d.router_calls.sum()), "cache_mismatch": int(d.cache_mismatch.sum()),
             "truncated": int(d.truncated.sum()), "http_errors": int(d.http_errors.sum()),
             "router_fallback": int(d.router_fallback.sum())}
    # reproduction check: T0 runs vs followup_v3 dev50 (same code path for always_A/always_B/follow_R0==L0)
    v3 = collections.defaultdict(list)
    for l in open(ROOT / "followup_v3/results/runs/dev50/calls.jsonl"):
        c = json.loads(l)
        v3[(c["session_id"], c["context"], c["policy"])].append(c["assistant_text"])
    mp = {"always_A": "always_A", "always_B": "always_B", "follow_R0": "L0"}
    by = collections.defaultdict(list)
    for c in sorted(C, key=lambda c: c["call"]):
        if c["schema"] == "T0" and c["policy"] in mp:
            by[(c["session_id"], c["context"], mp[c["policy"]])].append(c["assistant_text"])
    same = {k: by[k] == v3[k] for k in by}
    audit["T0_reproduces_v3_text"] = {"runs": len(same), "identical": sum(same.values()),
                                      "differ": [list(k) for k, v in same.items() if not v]}
    json.dump(audit, open(OUT / "audit.json", "w"), indent=1)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(json.dumps(audit, indent=1))
        print(summ.round(3).to_string())
        print(pr[["context", "a", "b", "n", "d_q", "d_q_ci", "d_S", "d_latency_s", "d_latency_s_ci", "d_format_errors",
                  "d_budget_exhausted_rate"]].round(3).to_string())


if __name__ == "__main__":
    main()
