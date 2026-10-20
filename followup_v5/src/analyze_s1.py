"""Stage 1 (S1) analysis (CPU) -> followup_v5/results/stage1/analysis/
Current request = the request in which the first repeat happened (branched request). Remaining = requests after it.
'Recoverable by B' = any of a1-a3 answers the current request correctly (ex-post label, analysis only).
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import V5  # noqa: E402

RD = V5 / "results/stage1/run"
OUT = V5 / "results/stage1/analysis"


def boot(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (None, None)
    m = np.random.default_rng(0).choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = [json.loads(l) for l in open(RD / "sessions.jsonl")]
    C = [json.loads(l) for l in open(RD / "calls.jsonl")]
    RA = [json.loads(l) for l in open(RD / "restore_audit.jsonl")]
    plan = {s["state_id"]: s for s in json.load(open(V5 / "configs/s1_states.json"))["states"]}
    by = collections.defaultdict(list)
    for c in C:
        by[(c["state_id"], c["branch"], c["rep"])].append(c)
    rows, texts = [], {}
    for s in S:
        k0 = s["state_request"]
        cs = sorted(by[(s["state_id"], s["branch"], s["rep"])], key=lambda c: c["call"])
        cur = [c for c in cs if c["request"] == k0]
        rest = [c for c in cs if c["request"] > k0]
        ct = [c["assistant_text"] for c in cur]
        rerep = any(a == b for a, b in zip(ct, ct[1:]))
        reqs = s["requests"]
        texts[(s["state_id"], s["branch"], s["rep"])] = [c["assistant_text"] for c in cs]
        rows.append({"state_id": s["state_id"], "split": s["split"], "context": plan[s["state_id"]]["context"],
                     "branch": s["branch"], "rep": s["rep"], "cur_correct": float(reqs[k0]["correct"]),
                     "cur_submitted": reqs[k0]["submitted"], "cur_calls": len(cur),
                     "cur_latency_s": sum(c["call_latency_s"] for c in cur),
                     "cur_tool_errors": sum(1 for c in cur for t in c["tools"] if t["error"]), "cur_rerepeat": rerep,
                     "cur_out_tokens": sum(c["completion_tokens"] for c in cur),
                     "first_hit": cur[0]["cached_tokens"] if cur else None, "first_prompt": cur[0]["prompt_tokens"] if cur else None,
                     "n_rem_requests": len(reqs) - k0 - 1,
                     "q_rem": float(np.mean([r["correct"] for r in reqs[k0 + 1:]])) if len(reqs) > k0 + 1 else None,
                     "S": float(s["S"]), "prev_all_correct": all(s["prev_requests_correct"]),
                     "J_s": s["latency_s"], "n_calls": s["n_calls"], "rest_latency_s": sum(c["call_latency_s"] for c in rest)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "branch_runs.csv", index=False)
    d0 = d[d.rep == 0]
    rec = d0[d0.branch.isin(["a1", "a2", "a3"])].groupby("state_id").cur_correct.max()
    d0 = d0.assign(recoverable_by_B=d0.state_id.map(rec).astype(bool))
    summ = d0.groupby(["branch"]).agg(n=("state_id", "size"), cur_correct=("cur_correct", "mean"), cur_calls=("cur_calls", "mean"),
                                      cur_latency_s=("cur_latency_s", "mean"), cur_tool_errors=("cur_tool_errors", "mean"),
                                      cur_rerepeat=("cur_rerepeat", "mean"), cur_out_tokens=("cur_out_tokens", "mean"),
                                      q_rem=("q_rem", "mean"), S=("S", "mean"), J_s=("J_s", "mean")).reset_index()
    summ.to_csv(OUT / "summary_by_branch.csv", index=False)
    by_layer = d0.groupby(["split", "context", "branch"]).agg(n=("state_id", "size"), cur_correct=("cur_correct", "mean"),
                                                               cur_latency_s=("cur_latency_s", "mean"), q_rem=("q_rem", "mean"),
                                                               J_s=("J_s", "mean")).reset_index()
    by_layer.to_csv(OUT / "summary_by_layer.csv", index=False)
    by_rec = d0.groupby(["recoverable_by_B", "branch"]).agg(n=("state_id", "size"), cur_correct=("cur_correct", "mean"),
                                                             cur_latency_s=("cur_latency_s", "mean"), q_rem=("q_rem", "mean"),
                                                             J_s=("J_s", "mean")).reset_index()
    by_rec.to_csv(OUT / "summary_by_recoverable.csv", index=False)
    # paired vs a0 and vs a1
    piv = {m: d0.pivot_table(index="state_id", columns="branch", values=m) for m in ("cur_correct", "cur_latency_s", "q_rem", "S", "J_s")}
    pr = []
    for a, b in [("a1", "a0"), ("a2", "a0"), ("a3", "a0"), ("a4", "a0"), ("a2", "a1"), ("a3", "a1")]:
        r = {"a": a, "b": b}
        for m, P in piv.items():
            x = (P[a] - P[b]).dropna()
            r[f"d_{m}"] = float(x.mean()); r[f"d_{m}_ci"] = boot(x); r["n"] = len(x)
        x = (piv["cur_correct"][a] - piv["cur_correct"][b]).dropna()
        r["n_better"], r["n_worse"] = int((x > 0).sum()), int((x < 0).sum())
        pr.append(r)
    pr = pd.DataFrame(pr)
    pr.to_csv(OUT / "paired.csv", index=False)
    # a0 vs a4 identity of outputs
    same04 = sum(texts.get((sid, "a0", 0)) == texts.get((sid, "a4", 0)) for sid in d0.state_id.unique())
    # replicate reproducibility
    reps = [(k[0], k[1]) for k in texts if k[2] == 1]
    same_rep = sum(texts[(a, b, 1)] == texts[(a, b, 0)] for a, b in reps)
    ra = pd.DataFrame(RA)
    audit = {"branch_runs": len(S), "states": int(d.state_id.nunique()), "calls": len(C),
             "restore_hash_ok": int(ra.prompt_hash_ok.sum()), "restore_rows": len(ra), "first_hit_ok": int(ra.first_hit_ok.sum()),
             "cache_mismatch": sum(c["pred_hit"][c["chosen"]] != c["cached_tokens"] for c in C),
             "truncated": sum(c["finish"] == "length" for c in C), "http_errors": sum(bool(c["error"]) for c in C),
             "router_fallback": sum(bool(c["router"] and c["router"]["parse_error"]) for c in C),
             "replicate_pairs": len(reps), "replicate_identical_text": same_rep,
             "a0_a4_identical_text_states": int(same04), "states_recoverable_by_B": int(rec.sum()), "states_total": int(len(rec))}
    json.dump(audit, open(OUT / "audit.json", "w"), indent=1, default=float)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(json.dumps(audit, indent=1))
        print(summ.round(3).to_string())
        print(by_rec.round(3).to_string())
        print(by_layer.round(3).to_string())
        print(pr[["a", "b", "n", "d_cur_correct", "d_cur_correct_ci", "n_better", "n_worse", "d_cur_latency_s", "d_q_rem", "d_J_s"]].round(3).to_string())


if __name__ == "__main__":
    main()
