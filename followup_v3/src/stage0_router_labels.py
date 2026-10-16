"""Stage 0-3 (cont.): router margin vs 'B needed' labels.

    python stage0_router_labels.py score     (needs R server: scores the 40 v2 branch states, active=A)
    python stage0_router_labels.py analyze   (CPU)
Same-state labels only (design.md stage 0-3):
  - v2 branch (40 points): next-call pass of stayA vs switchB from the identical restored state.
  - v1 initial-choice (10 points): turn-0 A vs B on identical input (router state = logged P4 turn 0).
Different-path proxy (report.md section 4) is reported separately and marked as proxy.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

V3 = Path(__file__).resolve().parents[1]
ROOT = V3.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(V3 / "src"))
from mrp import router  # noqa: E402
from mrp.client import Server  # noqa: E402
from stage0_router import conversation_at, score, h  # noqa: E402

OUT = V3 / "results/stage0"
SESS = {json.loads(l)["session_id"]: json.loads(l) for l in open(ROOT / "data/eval/eval.jsonl")}
OLD = {"short": ROOT / "results/evalA", "long": ROOT / "results/evalB_long_retain"}


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def score_branch_states():
    cfg = json.load(open(ROOT / "configs/models.json"))
    R = Server("R", cfg["models"]["R"], httpx.Client())
    end = R.tok.convert_tokens_to_ids("<|im_end|>")
    cands = {"SMALL": R.tok("SMALL", add_special_tokens=False)["input_ids"] + [end],
             "LARGE": R.tok("LARGE", add_special_tokens=False)["input_ids"] + [end]}
    pts = json.load(open(ROOT / "followup_v2/configs/branch_points.json"))
    T = defaultdict(list)
    for ctx, d in OLD.items():
        for l in open(d / "turns.jsonl"):
            t = json.loads(l)
            if t["policy"] == "P1":
                T[(t["session_id"], ctx)].append(t)
    out = []
    for i, p in enumerate(pts):
        ts = sorted(T[(p["session_id"], p["context"])], key=lambda t: t["turn"])
        msgs, table = conversation_at(SESS[p["session_id"]], ts, p["turn"], p["context"])
        ids = R.render_ids(router.build_messages(R.tok, msgs, table, "A"))
        sc = score(R, ids, cands)
        g = R.complete(ids, max_tokens=4)
        out.append({"point": i, **p, "ids_hash": h(ids), "margin": sc["LARGE"] - sc["SMALL"], "greedy": g["text"]})
    with open(OUT / "router_margin_v2_branch_states.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")


def analyze():
    rho = np.array(json.load(open(V3 / "configs/rho_mapping.json"))["dev_margins_sorted"])
    to_rho = lambda m: float(np.searchsorted(rho, m, side="right") / len(rho))
    res = {}
    # same-state set 1: v2 branch next call
    st = [json.loads(l) for l in open(OUT / "router_margin_v2_branch_states.jsonl")]
    pp = pd.read_csv(ROOT / "followup_v2/results/analysis_branch/paired_points.csv").set_index("point")
    rows = []
    for r in st:
        a, b = pp.loc[r["point"], "next_call_pass_stayA"], pp.loc[r["point"], "next_call_pass_switchB"]
        rows.append({**r, "nextA": a, "nextB": b, "needB": (a < 0.5) and (b >= 0.5), "rho": to_rho(r["margin"])})
    d1 = pd.DataFrame(rows)
    d1.to_csv(OUT / "router_label_v2_branch.csv", index=False)
    res["same_state_v2_branch_next_call"] = {"n": len(d1), "n_needB": int(d1.needB.sum()),
                                             "n_Aenough": int(((d1.nextA >= 0.5)).sum()),
                                             "auc_margin": auc(d1.margin, d1.needB),
                                             "R_greedy_LARGE": int((d1.greedy == "LARGE").sum())}
    # same-state set 2: v1 initial-choice (turn 0, short)
    mm = [json.loads(l) for l in open(OUT / "router_margin.jsonl")]
    t0 = {r["session_id"]: r["margin"] for r in mm if r["run"] == "evalA" and r["policy"] == "P4" and r["turn"] == 0}
    bi = pd.read_csv(V3.parent / "followup_v1/results/branch_init_summary/points.csv")
    bi["margin"] = bi.session_id.map(t0)
    bi["needB"] = (bi.pass_A < 0.5) & (bi.pass_B >= 0.5)
    res["same_state_v1_initial"] = {"n": len(bi), "n_needB": int(bi.needB.sum()), "auc_margin": auc(bi.margin, bi.needB)}
    # pooled same-state
    pool_m = list(d1.margin) + list(bi.margin)
    pool_y = list(d1.needB) + list(bi.needB)
    res["same_state_pooled"] = {"n": len(pool_y), "n_needB": int(sum(pool_y)), "auc_margin": auc(pool_m, pool_y)}
    # different-path proxy: P4 attempt-0 router calls, label from P1/P2 step outcomes of the same session
    prox = []
    for run, ctx in (("evalA", "short"), ("evalB_long_retain", "long")):
        S = {}
        for l in open(ROOT / "results" / run / "sessions.jsonl"):
            j = json.loads(l)
            S[(j["session_id"], j["policy"])] = j
        for r in mm:
            if r["run"] == run and r["policy"] == "P4" and r["attempt"] == 0:
                a = S[(r["session_id"], "P1")]["steps"][r["step"]]["pass"]
                b = S[(r["session_id"], "P2")]["steps"][r["step"]]["pass"]
                prox.append({"context": ctx, "margin": r["margin"], "needB_proxy": (not a) and b, "turn0": r["turn"] == 0})
    dp = pd.DataFrame(prox)
    res["different_path_proxy_P4"] = {c: {"n": int((dp.context == c).sum()), "n_needB": int(dp[dp.context == c].needB_proxy.sum()),
                                          "auc_margin": auc(dp[dp.context == c].margin, dp[dp.context == c].needB_proxy)}
                                      for c in ("short", "long")}
    res["different_path_proxy_P4"]["note"] = "proxy label from different policy paths (P1/P2), not same-state"
    allm = np.array([r["margin"] for r in mm])
    res["margin_distribution_all_calls"] = {"n": len(allm), "pct": dict(zip(["p1", "p10", "p25", "p50", "p75", "p90", "p99"],
                                                                           np.percentile(allm, [1, 10, 25, 50, 75, 90, 99]).round(3).tolist())),
                                            "share_abs_lt_1": float((abs(allm) < 1).mean())}
    json.dump(res, open(OUT / "router_label_auc.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    {"score": score_branch_states, "analyze": analyze}[sys.argv[1]]()
