"""followup_v4 section 1: recompute and audit followup_v3 dev50 results from raw logs (no new inference).

    python audit_v3.py      -> followup_v4/results/audit_v3/
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
V3 = ROOT / "followup_v3"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(V3 / "src"))
from mrp import router  # noqa: E402
from mrp.client import tokenizer  # noqa: E402
import agent_tasks as G  # noqa: E402
from gate1 import rebuild_prompts  # noqa: E402

OUT = V4 / "results/audit_v3"


def norm_err(e):
    e = re.sub(r"'[^']*'", "'X'", e)
    e = re.sub(r"\"[^\"]*\"", "\"X\"", e)
    return re.sub(r"\d+", "N", e)[:90]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = [json.loads(l) for l in open(V3 / "results/runs/dev50/sessions.jsonl")]
    C = [json.loads(l) for l in open(V3 / "results/runs/dev50/calls.jsonl")]
    res = {"runs": len(S), "unique_runs": len({(s["session_id"], s["policy"], s["context"]) for s in S}), "calls": len(C),
           "cache_pred_mismatch": sum(c["pred_hit"][c["chosen"]] != c["cached_tokens"] for c in C)}
    # long L0 vs always_B (paired by session)
    sd = pd.DataFrame([{"sid": s["session_id"], "pol": s["policy"], "ctx": s["context"], "q": s["q"], "S": float(s["S"]),
                        "lat": s["latency_s"], "wall": s["wall_s"]} for s in S])
    for ctx in ("short", "long"):
        a = sd[(sd.ctx == ctx) & (sd.pol == "L0")].set_index("sid")
        b = sd[(sd.ctx == ctx) & (sd.pol == "always_B")].set_index("sid")
        res[f"L0_minus_alwaysB_{ctx}"] = {"d_q": float((a.q - b.q).mean()), "d_S": float((a.S - b.S).mean()),
                                          "d_latency_s": float((a.lat - b.lat).mean()), "n": len(a),
                                          "L0": {"q": float(a.q.mean()), "S": float(a.S.mean()), "lat": float(a.lat.mean())},
                                          "always_B": {"q": float(b.q.mean()), "S": float(b.S.mean()), "lat": float(b.lat.mean())}}
    # C1-C2 disagreement recount
    d = pd.read_csv(V3 / "results/stage01/decision_points.csv")
    res["C1_C2_disagree_h1"] = {ctx: f"{int((g.C1_h1_pref != g.C2_h1_pref).sum())}/{len(g)}" for ctx, g in d.groupby("context")}
    # cost-representation errors (realized vs predicted output length), recomputed from the v3 offline table
    e = pd.read_csv(V3 / "results/stage01/live_call_cost_predictions.csv")
    errs = []
    for (ctx, m), g in e.groupby(["context", "model"]):
        for rep in ("C0", "C1", "C2", "C3"):
            errs.append({"context": ctx, "model": m, "rep": rep,
                         "mae_realized_y": float((g[f"{rep}_yreal"] - g.measured_total_s).abs().mean()),
                         "mae_yhat": float((g[f"{rep}_yhat"] - g.measured_total_s).abs().mean())})
    pd.DataFrame(errs).to_csv(OUT / "cost_errors_recomputed.csv", index=False)
    # tool errors: kinds, model, repetition within request
    rows = []
    for c in C:
        calls, _ = G.parse_tool_calls(c["assistant_text"])
        for (name, args), t in zip(calls, c["tools"]):
            if t["error"]:
                rows.append({"session_id": c["session_id"], "policy": c["policy"], "context": c["context"],
                             "request": c["request"], "call": c["call"], "model": c["chosen"], "tool": name,
                             "error": t["error"], "kind": norm_err(t["error"]), "args": json.dumps(args)[:300]})
    te = pd.DataFrame(rows)
    te.to_csv(OUT / "tool_errors.csv", index=False)
    n_tool = sum(len(c["tools"]) for c in C)
    kinds = te.groupby(["kind", "model"]).size().unstack(fill_value=0)
    kinds["total"] = kinds.sum(1)
    kinds = kinds.sort_values("total", ascending=False)
    kinds.to_csv(OUT / "tool_error_kinds.csv")
    tool_calls_by_model = collections.Counter(c["chosen"] for c in C for _ in c["tools"])
    res["tool_errors"] = {"n": len(te), "n_tool_calls": n_tool,
                          "by_model": {m: {"errors": int((te.model == m).sum()), "tool_calls": tool_calls_by_model[m],
                                           "rate": float((te.model == m).sum() / tool_calls_by_model[m])} for m in "AB"},
                          "top_kinds": kinds.head(8)["total"].to_dict()}
    # unknown column 'None': what argument produced it
    none = te[te.error.str.contains("unknown column 'None", regex=False)]  # env strips the trailing quote
    pat = collections.Counter()
    for a in none.args:
        try:
            j = json.loads(a)
        except json.JSONDecodeError:
            pat["args not parseable (log truncated at 300 chars)"] += 1
            continue
        conds = j.get("conditions") if isinstance(j, dict) else None
        if isinstance(conds, list) and conds and isinstance(conds[0], dict):
            pat["filter condition without 'column' key; keys=" + ",".join(sorted(conds[0]))] += 1
        else:
            pat["other: " + a[:60]] += 1
    res["unknown_column_None"] = {"n": len(none), "by_model": none.model.value_counts().to_dict(),
                                  "argument_patterns": dict(pat.most_common(8)),
                                  "examples": none.args.head(5).tolist()}
    # repetition: same normalized error repeated within the same request (same run)
    rep = te.groupby(["session_id", "policy", "context", "request", "kind"]).size()
    res["repetition"] = {"errors_in_runs_of_same_error_within_request": int(rep[rep > 1].sum()),
                         "error_request_groups": int(len(rep)), "groups_with_repeat": int((rep > 1).sum()),
                         "max_repeat": int(rep.max()),
                         "errors_if_counted_once_per_request_kind": int(len(rep))}
    req_fmt = collections.defaultdict(bool)
    for c in C:
        for t in c["tools"]:
            if t["error"]:
                req_fmt[(c["session_id"], c["policy"], c["context"], c["request"])] = True
    n_req = sum(len(s["requests"]) for s in S)
    res["requests_with_any_tool_error"] = {"n": sum(req_fmt.values()), "of": n_req,
                                           "by_policy_context": {}}
    for s in S:
        k = f"{s['policy']}|{s['context']}"
        res["requests_with_any_tool_error"]["by_policy_context"].setdefault(k, [0, 0])
        res["requests_with_any_tool_error"]["by_policy_context"][k][0] += sum(
            req_fmt[(s["session_id"], s["policy"], s["context"], r["request"])] for r in s["requests"])
        res["requests_with_any_tool_error"]["by_policy_context"][k][1] += len(s["requests"])
    # router input audit (R0): roles shown and whether the current user request is visible
    rcfg = json.load(open(ROOT / "configs/models.json"))["models"]["R"]
    rtok = tokenizer(rcfg["hf_id"], rcfg["revision"], rcfg.get("local_path"))
    acfg = json.load(open(ROOT / "configs/models.json"))["models"]["A"]
    atok_srv = type("S", (), {"tok": tokenizer(acfg["hf_id"], acfg["revision"])})()
    sessions = {json.loads(l)["session_id"]: json.loads(l) for l in open(V3 / "data/dev.jsonl")}
    by = collections.defaultdict(list)
    for c in C:
        if c["policy"] == "L0":
            by[(c["session_id"], c["context"])].append(c)
    aud = []
    for (sid, ctx), cs in by.items():
        cs = sorted(cs, key=lambda c: c["call"])
        for c in cs:
            msgs, _ = rebuild_prompts(atok_srv, sessions[sid], ctx, cs, c["call"])
            rm = router.build_messages(rtok, msgs, "", c["active_before"])
            body = rm[1]["content"]
            last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user" and "[Request" in m["content"])
            req_text = msgs[last_user]["content"]
            visible_window = msgs[1:-1][-router.HIST_MESSAGES:]
            in_window = any(m is msgs[last_user] for m in visible_window) or msgs[-1] is msgs[last_user]
            truncated = in_window and req_text not in body
            aud.append({"session_id": sid, "context": ctx, "call": c["call"], "current_role": msgs[-1]["role"],
                        "request_visible": in_window and not truncated, "request_dropped_by_window": not in_window,
                        "request_truncated": truncated, "tool_shown_as_ASSISTANT": "ASSISTANT: {" in body or
                        ("ASSISTANT:" in body and any(m["role"] == "tool" for m in visible_window)),
                        "router_input_tokens": c["router"]["input_tokens"]})
    ra = pd.DataFrame(aud)
    ra.to_csv(OUT / "router_input_audit.csv", index=False)
    res["router_input_R0"] = {"calls": len(ra), "current_message_role": ra.current_role.value_counts().to_dict(),
                              "request_dropped_by_window": int(ra.request_dropped_by_window.sum()),
                              "request_truncated_in_window": int(ra.request_truncated.sum()),
                              "request_visible": int(ra.request_visible.sum()),
                              "tool_messages_labeled_ASSISTANT": "router.py labels every non-user role as ASSISTANT; "
                                                                 "tool results appear as 'ASSISTANT: {...}'",
                              "max_input_tokens": int(ra.router_input_tokens.max())}
    # latency definition: sum of components vs wall clock
    comp = collections.defaultdict(float)
    for c in C:
        comp[(c["session_id"], c["policy"], c["context"])] += c["router_s"] + c["render_s"] + c["total_s"] + c["tool_exec_s"]
    diffs = [abs(comp[(s["session_id"], s["policy"], s["context"])] - s["latency_s"]) for s in S]
    gap = [s["wall_s"] - s["latency_s"] for s in S]
    res["latency_definition"] = {"latency_s = sum(router_s + render_s + model call total_s + tool_exec_s)": True,
                                 "max_abs_recompute_diff_s": float(max(diffs)),
                                 "wall_minus_latency_s_median": float(np.median(gap)), "wall_minus_latency_s_max": float(max(gap)),
                                 "note": "wall includes Prometheus scraping and bookkeeping; isolation reset is outside both",
                                 "same_as_previous_experiments": "same component-sum definition as v1/v2 (no gate/evict/verifier terms in v3)"}
    json.dump(res, open(OUT / "audit_v3.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    print(kinds.head(12).to_string())


if __name__ == "__main__":
    main()
