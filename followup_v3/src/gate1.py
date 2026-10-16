"""Stage 1-3: cost-representation decision disagreement at L0 decision points + gate 1 + paired check.

    python gate1.py analyze                     (CPU)
    python gate1.py select-paired               (CPU; writes configs/paired_points.json before measuring)
    python gate1.py measure-paired              (needs A, B servers; under flock)
    python gate1.py gate                        (CPU; final gate table)
Cost-only preferences (no quality term; not a policy decision): follow iff C_hat(rec) < C_hat(cur).
"""
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

V3 = Path(__file__).resolve().parents[1]
ROOT = V3.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(V3 / "src"))
from mrp.client import CacheTracker, Server  # noqa: E402
import agent_tasks as G  # noqa: E402
from profile_v3 import CostModels  # noqa: E402
from agent_runner import render, MAX_TOKENS  # noqa: E402

OUT = V3 / "results/stage01"
YHAT = {"A": 35.0, "B": 36.8}
NHAT = {"short": 9.0, "long": 11.8}
DELTA = {"short": 98.0, "long": 378.0}
REPS = ["C0", "C1", "C2", "C3"]
PAIRS = [("C0", "C1"), ("C0", "C2"), ("C0", "C3"), ("C1", "C2"), ("C1", "C3"), ("C2", "C3")]


def chat(cm, rep, m, ell, c, ctx, h):
    first = cm.predict(rep, m, ell, c, YHAT[m])
    if h == "1":
        return first
    cont = cm.predict(rep, m, ell + DELTA[ctx], ell, YHAT[m])
    return first + (NHAT[ctx] - 1) * cont


def load_calls(run="dev50"):
    return [json.loads(l) for l in open(V3 / "results/runs" / run / "calls.jsonl")]


def decision_points(C):
    return [c for c in C if c["policy"] == "L0" and c["active_before"] is not None
            and c["router"]["choice"] != c["active_before"]]


def analyze():
    cm = CostModels.load(OUT / "profile/cost_models.json")
    C = load_calls()
    dps = decision_points(C)
    rows = []
    for c in dps:
        cur, rec, ell, ctx = c["active_before"], c["router"]["choice"], c["prompt_tokens"], c["context"]
        r = {"session_id": c["session_id"], "context": ctx, "call": c["call"], "cur": cur, "rec": rec,
             "direction": f"{cur}->{rec}", "ell": ell, "c_cur": c["pred_hit"][cur], "c_rec": c["pred_hit"][rec],
             "realized_out": c["completion_tokens"], "realized_total_s": c["total_s"]}
        for rep in REPS:
            for h in ("1", "N"):
                kc, fc = chat(cm, rep, cur, ell, c["pred_hit"][cur], ctx, h), chat(cm, rep, rec, ell, c["pred_hit"][rec], ctx, h)
                r[f"{rep}_h{h}_keep"], r[f"{rep}_h{h}_follow"] = kc, fc
                r[f"{rep}_h{h}_pref"] = "follow" if fc < kc else "keep"
                # realized-output variant (diagnostic only; future length is not a decision input)
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "decision_points.csv", index=False)
    n_calls = defaultdict(int)
    for c in C:
        if c["policy"] == "L0":
            n_calls[c["context"]] += 1
    out = []
    for ctx in ("short", "long"):
        for h in ("1", "N"):
            for direction in ("all", "A->B", "B->A"):
                sub = d[d.context == ctx] if direction == "all" else d[(d.context == ctx) & (d.direction == direction)]
                for a, b in PAIRS:
                    dis = (sub[f"{a}_h{h}_pref"] != sub[f"{b}_h{h}_pref"]).sum()
                    out.append({"context": ctx, "h": h, "direction": direction, "pair": f"{a}-{b}", "n_points": len(sub),
                                "n_disagree": int(dis), "rate": dis / len(sub) if len(sub) else None,
                                "n_L0_calls": n_calls[ctx], "rate_of_all_calls": dis / n_calls[ctx]})
                for rep in REPS:
                    out.append({"context": ctx, "h": h, "direction": direction, "pair": f"{rep}:follow_share",
                                "n_points": len(sub), "rate": float((sub[f"{rep}_h{h}_pref"] == "follow").mean()) if len(sub) else None})
    g = pd.DataFrame(out)
    g.to_csv(OUT / "gate_disagreement.csv", index=False)
    plot_plane(cm, d)
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(g[g.pair.isin(["C1-C2", "C1-C3", "C0-C1", "C2-C3"]) & (g.direction == "all")].round(3).to_string())
        print(g[g.pair.str.contains("follow_share") & (g.direction != "all")].round(3).to_string())


def plot_plane(cm, d):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    ells = np.linspace(500, 24000, 60)
    for ax, (cur, rec) in zip(axs, (("B", "A"), ("A", "B"))):
        sub = d[(d.cur == cur) & (d.rec == rec)]
        ax.scatter(sub.ell, sub.realized_out, s=10, alpha=0.5, c=["#1f77b4" if x == "short" else "#d62728" for x in sub.context],
                   label="decision points (realized output, for display only)")
        for rep, col in zip(REPS, ("#999999", "#2ca02c", "#9467bd", "#ff7f0e")):
            ys = []
            for ell in ells:  # break-even output length: C(rec, cold) == C(cur, warm=ell-100)
                lo, hi, found = 1, 600, None
                f = lambda y: cm.predict(rep, rec, ell, 0.0, y) - cm.predict(rep, cur, ell, max(ell - 100, 0), y)
                yy = np.arange(1, 600)
                vals = np.array([f(y) for y in yy])
                idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
                ys.append(yy[idx[0]] if len(idx) else np.nan)
            ax.plot(ells, ys, color=col, label=f"{rep} break-even (target cold, current warm)")
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("output length (tokens)")
        ax.set_ylim(0, 300)
        ax.set_title(f"current {cur} -> recommended {rec}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_breakeven_plane.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- paired measurement
def select_paired():
    d = pd.read_csv(OUT / "decision_points.csv")
    d["c1_vs_c23"] = (d["C1_h1_pref"] != d["C2_h1_pref"]) | (d["C1_h1_pref"] != d["C3_h1_pref"])
    d["gap"] = (d["C2_h1_keep"] - d["C2_h1_follow"]).abs()
    pts = []
    for (ctx, dirn), g in d.groupby(["context", "direction"]):
        g = g.copy()
        g["key"] = [hashlib.sha256(f"v3pair|{s}|{c}|{ctx}".encode()).hexdigest() for s, c in zip(g.session_id, g.call)]
        dis = g[g.c1_vs_c23].sort_values("key")
        rest = g[~g.c1_vs_c23].sort_values("gap")
        pick = pd.concat([dis.head(6), rest.head(max(0, 6 - len(dis)))])
        pts += pick[["session_id", "context", "call", "cur", "rec", "direction", "c1_vs_c23"]].to_dict("records")
    json.dump({"rule": "per (context, direction): up to 6 points; C1 vs C2/C3 disagreement points first (hash order), "
                       "otherwise smallest |C2 keep-follow| gap; chosen before measuring", "points": pts},
              open(V3 / "configs/paired_points.json", "w"), indent=1)
    print(len(pts), pd.DataFrame(pts).groupby(["context", "direction", "c1_vs_c23"]).size())


def replay(sess, context, calls_upto):
    """Rebuild messages exactly as agent_runner did, by re-executing logged tool calls in a fresh Env."""
    env = G.Env(sess, context)
    msgs = [{"role": "system", "content": G.SYSTEM}]
    notice, k_prev, sub = "", -1, False
    by_req = defaultdict(list)
    for c in calls_upto:
        by_req[c["request"]].append(c)
    return env, msgs, by_req


def rebuild_prompts(srv, sess, context, cs, upto_call):
    env = G.Env(sess, context)
    msgs = [{"role": "system", "content": G.SYSTEM}]
    prompts, notice = {}, ""
    cs = sorted(cs, key=lambda c: c["call"])
    for k in range(len(sess["requests"])):
        msgs.append({"role": "user", "content": notice + G.request_message(sess, k)})
        rc = [c for c in cs if c["request"] == k]
        submitted = False
        for c in rc:
            ids = render(srv, msgs)
            prompts[c["call"]] = ids
            if c["call"] == upto_call:
                return msgs, prompts
            msgs.append({"role": "assistant", "content": c["assistant_text"]})
            calls, perr = G.parse_tool_calls(c["assistant_text"])
            for name, args in calls[: G.MAX_TOOL_CALLS_PER_TURN]:
                res = env.call(name, args)
                msgs.append({"role": "tool", "content": json.dumps(res)})
                if name == "submit" and "error" not in res:
                    submitted = True
                    break
            if not calls:
                msgs.append({"role": "user", "content": "[Environment] No valid tool call found"
                             + (f" ({perr[0]})" if perr else "") + ". Respond with a tool call."})
        notice = "" if submitted else "[Environment] The call budget for the previous request was exhausted.\n"
    raise ValueError("call not found")


def measure_paired():
    cfg = json.load(open(ROOT / "configs/models.json"))
    http = httpx.Client()
    srv = {m: Server(m, cfg["models"][m], http) for m in ("A", "B")}
    for s in srv.values():
        w = s.render_ids([{"role": "user", "content": "hi"}])
        for _ in range(3):
            s.complete(w, max_tokens=16)
    pts = json.load(open(V3 / "configs/paired_points.json"))["points"]
    sessions = {json.loads(l)["session_id"]: json.loads(l) for l in open(V3 / "data/dev.jsonl")}
    C = load_calls()
    by = defaultdict(list)
    for c in C:
        if c["policy"] == "L0":
            by[(c["session_id"], c["context"])].append(c)
    fh = open(OUT / "paired_measurements.jsonl", "a")
    for i, p in enumerate(pts):
        cs = sorted(by[(p["session_id"], p["context"])], key=lambda c: c["call"])
        msgs, prompts = rebuild_prompts(srv["A"], sessions[p["session_id"]], p["context"], cs, p["call"])
        logged = next(c for c in cs if c["call"] == p["call"])
        ids = prompts[p["call"]]
        ok = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16] == logged["prompt_sha"]
        held = {"A": [], "B": []}
        for c in cs:
            if c["call"] >= p["call"]:
                break
            held[c["chosen"]].append(prompts[c["call"]])
        tr = CacheTracker()
        for m in "AB":
            for x in held[m]:
                tr.add(m, x)
        order = [("keep", p["cur"]), ("follow", p["rec"])]
        sched = []
        for r in range(3):
            sched += order if r % 2 == 0 else order[::-1]
        for r, (act, m) in enumerate(sched):
            S = srv[m]
            S.reset_cache(); time.sleep(0.2)
            t0 = time.perf_counter()
            for x in held[m]:
                S.complete(x, max_tokens=1)
            prep = time.perf_counter() - t0
            time.sleep(0.3)
            c = S.complete(ids, max_tokens=MAX_TOKENS)
            rec = {"point": i, **p, "action": act, "model": m, "rep": r // 2, "reconstruct_ok": ok,
                   "expected_hit": tr.predicted_hit(m, ids), "hit": c["cached_tokens"], "total_s": c["total_s"],
                   "ttft_s": c["ttft_s"], "completion_tokens": c["completion_tokens"], "prep_s": prep,
                   "text_sha": hashlib.sha256(c["text"].encode()).hexdigest()[:16]}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(i, p["session_id"], p["context"], p["direction"], "ok", ok, flush=True)


def gate():
    g = pd.read_csv(OUT / "gate_disagreement.csv")
    S = [json.loads(l) for l in open(V3 / "results/runs/dev50/sessions.jsonl")]
    sd = pd.DataFrame([{"session_id": s["session_id"], "policy": s["policy"], "context": s["context"], "q": s["q"], "S": s["S"]} for s in S])
    res = {}
    cell = g[g.pair.isin(["C1-C2", "C1-C3"]) & (g.direction == "all")]
    ok_a = cell[(cell.rate >= 0.15) & (cell.n_points >= 30)]
    res["a_C1_vs_C2C3_disagreement"] = {"max_rate": float(cell.rate.max()), "cells_passing": ok_a[["context", "h", "pair", "n_points", "rate"]].to_dict("records"),
                                        "pass": len(ok_a) > 0}
    c0 = g[g.pair.isin(["C0-C1", "C0-C2", "C0-C3"]) & (g.direction == "all")]
    res["C0_pairs_reported_separately"] = c0[["context", "h", "pair", "n_points", "rate"]].to_dict("records")
    for ctx in ("short", "long"):
        a = sd[(sd.context == ctx) & (sd.policy == "always_A")].set_index("session_id")
        b = sd[(sd.context == ctx) & (sd.policy == "always_B")].set_index("session_id")
        res[f"b_S_gap_{ctx}"] = {"S_B": float(b.S.mean()), "S_A": float(a.S.mean()), "gap": float(b.S.mean() - a.S.mean()),
                                 "pass": float(b.S.mean() - a.S.mean()) >= 0.15}
        ae = (a.q >= b.q.reindex(a.index))
        res[f"c_A_enough_{ctx}"] = {"rate": float(ae.mean()), "both_zero": float(((a.q == 0) & (b.q.reindex(a.index) == 0)).mean()),
                                    "pass": 0.2 <= float(ae.mean()) <= 0.6}
    pm = OUT / "paired_measurements.jsonl"
    if pm.exists():
        P = pd.DataFrame([json.loads(l) for l in open(pm)])
        res["paired"] = {"rows": len(P), "matched_state": int((P.hit == P.expected_hit).sum()),
                         "reconstruct_ok": bool(P.reconstruct_ok.all()),
                         "varying_output": int((P.groupby(["point", "action"]).text_sha.nunique() > 1).sum())}
    res["gate1_pass"] = res["a_C1_vs_C2C3_disagreement"]["pass"] and all(res[f"b_S_gap_{c}"]["pass"] for c in ("short", "long")) \
        and all(res[f"c_A_enough_{c}"]["pass"] for c in ("short", "long"))
    res["note"] = "(b),(c) evaluated per context; gate requires each context to pass (stricter reading)"
    json.dump(res, open(OUT / "gate1.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    {"analyze": analyze, "select-paired": select_paired, "measure-paired": measure_paired, "gate": gate}[sys.argv[1]]()
