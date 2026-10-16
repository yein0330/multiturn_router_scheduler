"""Stage 0-2: exploratory "is A enough?" signals (not training a policy; results are [hypothesis]).

    python stage0_signal.py
Timepoints: initial-pre-call vs after-first-A (design.md stage 0-2).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

V3 = Path(__file__).resolve().parents[1]
ROOT = V3.parent
OUT = V3 / "results/stage0"
import sys
sys.path.insert(0, str(ROOT / "src"))
from mrp.client import tokenizer  # noqa: E402

CFG = json.load(open(ROOT / "configs/models.json"))["models"]["A"]
TOK = tokenizer(CFG["hf_id"], CFG["revision"])
OPS = ["count", "sum_qty", "max_price_id", "distinct_cat2", "mean_qty", "group_argmax_rev", "median_by_cat2",
       "top3_multikey", "rate_threshold", "running_balance", "longest_day_streak", "second_cat2"]


def load_pairs(run_dirs, sess_files, ctx_filter):
    sess = {}
    for f in sess_files:
        for l in open(f):
            j = json.loads(l)
            sess[j["session_id"]] = j
    S, T0 = {}, {}
    for d in run_dirs:
        for l in open(d / "sessions.jsonl"):
            j = json.loads(l)
            if j["policy"] in ("P1", "P2") and j["context"] == ctx_filter and j["cache_mode"] == "retain":
                S[(j["session_id"], j["policy"])] = j
        for l in open(d / "turns.jsonl"):
            j = json.loads(l)
            if j["policy"] == "P1" and j["turn"] == 0 and j["context"] == ctx_filter and j["cache_mode"] == "retain":
                T0[j["session_id"]] = j
    rows = []
    for sid, s in sess.items():
        if (sid, "P1") not in S or (sid, "P2") not in S or sid not in T0:
            continue
        a, b = S[(sid, "P1")], S[(sid, "P2")]
        st0 = s["steps"][0]
        op = st0["spec"]["agg"]["name"]
        t0 = T0[sid]
        v = (t0.get("verifier") or {}).get("status", "none")
        rows.append({"session_id": sid, "scenario": s["scenario"], "context": ctx_filter,
                     "qA": a["step_pass_rate"], "qB": b["step_pass_rate"], "latA": a["latency_s"], "latB": b["latency_s"],
                     "L_lex": None,  # filled for eval from the six-policy set; undefined on dev
                     "L_ge": a["step_pass_rate"] >= b["step_pass_rate"],
                     "both_zero": a["step_pass_rate"] == 0 and b["step_pass_rate"] == 0,
                     "op": op, "req_tokens": len(TOK(st0["request"], add_special_tokens=False)["input_ids"]),
                     "firstA_tokens": t0["call"]["completion_tokens"], "firstA_verifier": v,
                     "firstA_cost_s": t0["turn_latency_s"]})
    return pd.DataFrame(rows)


def features(df, timepoint):
    X = [np.ones(len(df)), (df.req_tokens.values - 60) / 20]
    names = ["bias", "req_tokens_z"]
    for op in OPS:
        X.append((df.op == op).astype(float).values)
        names.append("op=" + op)
    if timepoint == "after_first_A":
        X.append((df.firstA_tokens.values - 100) / 50)
        names.append("firstA_tokens_z")
        for v in ("FAIL", "TOOL_ERROR", "PASS"):
            X.append((df.firstA_verifier == v).astype(float).values)
            names.append("firstA_verifier=" + v)
    return np.vstack(X).T, names


def fit_logreg(X, y, C=1.0, iters=200):
    w = np.zeros(X.shape[1])
    lam = 1.0 / C
    for _ in range(iters):  # Newton / IRLS with L2 (bias not penalized)
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + lam * np.r_[0, w[1:]]
        H = X.T @ (X * (p * (1 - p))[:, None]) + lam * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
    return w


def auc(score, y):
    y = np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def calib(p, y, bins=(0, 0.33, 0.67, 1.0001)):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        out.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": int(m.sum()),
                    "mean_pred": float(p[m].mean()) if m.any() else None,
                    "obs_rate": float(np.mean(y[m])) if m.any() else None})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev_runs = [ROOT / "results/dev_smoke", ROOT / "results/dev_calib_sessions"]
    dev_files = [ROOT / "data/dev/smoke.jsonl", ROOT / "data/dev/calib_sessions.jsonl"]
    ev_runs = {"short": [ROOT / "results/evalA"], "long": [ROOT / "results/evalB_long_retain"]}
    ev = {c: load_pairs(ev_runs[c], [ROOT / "data/eval/eval.jsonl"], c) for c in ("short", "long")}
    # L_lex = AAAAA is the lexicographic ex-post best within the six v2 schedules (eval only; dev has only P1/P2)
    best6 = pd.read_csv(ROOT / "followup_v2/results/analysis_main/ex_post_best_within_six.csv")
    for c in ("short", "long"):
        m = best6[best6.context == c].set_index("session_id").lexicographic_best
        ev[c]["L_lex"] = ev[c].session_id.map(lambda s: m[s] == "AAAAA")
    dev = load_pairs(dev_runs, dev_files, "short")
    report = {"n": {"dev_short": len(dev), "eval_short": len(ev["short"]), "eval_long": len(ev["long"])}}
    for name, d in [("dev_short", dev), ("eval_short", ev["short"]), ("eval_long", ev["long"])]:
        report.setdefault("class_rates", {})[name] = {"L_lex": None if name == "dev_short" else float(d.L_lex.mean()),
                                                      "L_ge": float(d.L_ge.mean()),
                                                      "both_zero": float(d.both_zero.mean())}
    pd.concat([dev.assign(split="dev"), ev["short"].assign(split="eval"), ev["long"].assign(split="eval")]).to_csv(
        OUT / "signal_dataset.csv", index=False)
    # univariate associations on eval
    uni = []
    for ctx, d in ev.items():
        for lab in ("L_lex", "L_ge"):
            y = d[lab].values
            for f in ("req_tokens", "firstA_tokens"):
                uni.append({"context": ctx, "label": lab, "feature": f, "timepoint":
                            "initial" if f == "req_tokens" else "after_first_A", "auc_higher_means_A_enough":
                            auc(d[f].values.astype(float), y), "n": len(d), "pos": int(y.sum())})
            for op, g in d.groupby("op"):
                uni.append({"context": ctx, "label": lab, "feature": "op=" + op, "timepoint": "initial",
                            "n": len(g), "pos": int(g[lab].sum()), "rate": float(g[lab].mean())})
            for v, g in d.groupby("firstA_verifier"):
                uni.append({"context": ctx, "label": lab, "feature": "firstA_verifier=" + v,
                            "timepoint": "after_first_A", "n": len(g), "pos": int(g[lab].sum()), "rate": float(g[lab].mean())})
    pd.DataFrame(uni).to_csv(OUT / "signal_univariate.csv", index=False)
    # multivariate: short = fit dev, eval on eval; long = scenario LOO on eval
    res = []
    for lab in ("L_lex", "L_ge"):
        for tp in ("initial", "after_first_A"):
            if lab == "L_lex":  # no dev label: scenario LOO on eval for both contexts
                for ctx in ("short", "long"):
                    d = ev[ctx]
                    ps = np.zeros(len(d))
                    for sc in d.scenario.unique():
                        tr, te = d.scenario != sc, d.scenario == sc
                        wl = fit_logreg(features(d[tr], tp)[0], d[tr][lab].values.astype(float))
                        ps[te.values] = 1 / (1 + np.exp(-features(d[te], tp)[0] @ wl))
                    y = d[lab].values.astype(float)
                    res.append({"context": ctx, "label": lab, "timepoint": tp, "protocol": "scenario leave-one-out on eval",
                                "n": len(y), "pos_rate": float(y.mean()), "auc": auc(ps, y),
                                "brier": float(np.mean((ps - y) ** 2)), "brier_base": float(np.mean((y.mean() - y) ** 2)),
                                "calibration": calib(ps, y)})
                continue
            Xd, names = features(dev, tp)
            w = fit_logreg(Xd, dev[lab].values.astype(float))
            Xe, _ = features(ev["short"], tp)
            p = 1 / (1 + np.exp(-Xe @ w))
            y = ev["short"][lab].values.astype(float)
            res.append({"context": "short", "label": lab, "timepoint": tp, "protocol": "fit dev(30) -> eval(50)",
                        "n": len(y), "pos_rate": float(y.mean()), "auc": auc(p, y), "brier": float(np.mean((p - y) ** 2)),
                        "brier_base": float(np.mean((y.mean() - y) ** 2)), "calibration": calib(p, y),
                        "coef": dict(zip(names, np.round(w, 3).tolist()))})
            d = ev["long"]
            ps = np.zeros(len(d))
            for sc in d.scenario.unique():
                tr, te = d.scenario != sc, d.scenario == sc
                Xtr, _ = features(d[tr], tp)
                wl = fit_logreg(Xtr, d[tr][lab].values.astype(float))
                Xte, _ = features(d[te], tp)
                ps[te.values] = 1 / (1 + np.exp(-Xte @ wl))
            y = d[lab].values.astype(float)
            res.append({"context": "long", "label": lab, "timepoint": tp, "protocol": "scenario leave-one-out on eval",
                        "n": len(y), "pos_rate": float(y.mean()), "auc": auc(ps, y), "brier": float(np.mean((ps - y) ** 2)),
                        "brier_base": float(np.mean((y.mean() - y) ** 2)), "calibration": calib(ps, y)})
    report["models"] = res
    report["after_first_A_cost_s_mean"] = {c: float(d.firstA_cost_s.mean()) for c, d in ev.items()}
    report["note"] = "[hypothesis] exploratory; small n; not used as a policy; op family is an observable proxy derived from the request text"
    json.dump(report, open(OUT / "signal_report.json", "w"), indent=1, default=float)
    for r in res:
        print(r["context"], r["label"], r["timepoint"], "n", r["n"], "pos", round(r["pos_rate"], 2), "AUC",
              None if r["auc"] is None else round(r["auc"], 3), "Brier", round(r["brier"], 3), "base", round(r["brier_base"], 3))
    print(json.dumps(report["class_rates"]))
    print(pd.DataFrame(uni).query("feature in ['req_tokens','firstA_tokens']").round(3).to_string())


if __name__ == "__main__":
    main()
