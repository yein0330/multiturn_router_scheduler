"""Part 3 summary. Next-call comparison only; not a counterfactual of the session or an optimal policy."""
import json
import sys
from pathlib import Path

import pandas as pd


def main(meas, out, kind):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([json.loads(l) for l in open(meas)])
    audit = {"rows": len(df), "points": int(df.point.nunique()), "reconstruct_ok_all": bool(df.reconstruct_ok.all()),
             "matched_state_rows": int(df.matched_state.sum()), "truncated": int((df.finish == "length").sum()),
             "errors": int(df.error.notna().sum())}
    # determinism of outputs across reps
    stab = df.groupby(["point", "model"]).text_sha.nunique()
    audit["points_models_with_varying_output"] = int((stab > 1).sum())
    pm = df.groupby(["point", "model"]).agg(session_id=("session_id", "first"), scenario=("scenario", "first"),
                                            policy=("policy", "first"), context=("context", "first"),
                                            cache=("cache", "first"), turn=("turn", "first"),
                                            matched=("matched_state", "all"), pass_rate=("pass", "mean"),
                                            pass_any=("pass", "max"), total_med=("total_s", "median"),
                                            total_min=("total_s", "min"), total_max=("total_s", "max"),
                                            ttft_med=("ttft_s", "median"), ct_med=("completion_tokens", "median"),
                                            hit=("hit", "first"), expected_hit=("expected_hit", "first"),
                                            prep_med=("prep_s", "median"), chat=("chat_total_s", "first"),
                                            router_choice=("router_choice", "first")).reset_index()
    w = pm.pivot(index="point", columns="model")
    P = pd.DataFrame({"session_id": w["session_id"]["A"], "scenario": w["scenario"]["A"], "policy": w["policy"]["A"],
                      "context": w["context"]["A"], "cache": w["cache"]["A"], "turn": w["turn"]["A"],
                      "matched_A": w["matched"]["A"], "matched_B": w["matched"]["B"],
                      "pass_A": w["pass_rate"]["A"], "pass_B": w["pass_rate"]["B"],
                      "lat_A": w["total_med"]["A"], "lat_B": w["total_med"]["B"],
                      "tok_A": w["ct_med"]["A"], "tok_B": w["ct_med"]["B"],
                      "hit_A": w["hit"]["A"], "hit_B": w["hit"]["B"],
                      "chat_A": w["chat"]["A"], "chat_B": w["chat"]["B"],
                      "router_choice": w["router_choice"]["A"]})
    P["pred_diff_B_minus_A"] = P.chat_B - P.chat_A
    P["meas_diff_B_minus_A"] = P.lat_B - P.lat_A
    P["pred_faster"] = P.pred_diff_B_minus_A.map(lambda x: "A" if x > 0 else "B")
    P["meas_faster"] = P.meas_diff_B_minus_A.map(lambda x: "A" if x > 0 else "B")
    P["faster_correct"] = P.pred_faster == P.meas_faster
    def cls(r):
        a, b = r.pass_A >= 0.5, r.pass_B >= 0.5
        return "both_pass" if a and b else "both_fail" if not a and not b else "only_A" if a else "only_B"
    P["quality_class"] = P.apply(cls, axis=1)
    P.to_csv(out / "points.csv", index_label="point")
    pm.to_csv(out / "point_model.csv", index=False)
    g = P.groupby("quality_class").agg(n=("session_id", "count"), meas_diff_B_minus_A=("meas_diff_B_minus_A", "mean"),
                                       pred_diff_B_minus_A=("pred_diff_B_minus_A", "mean"),
                                       faster_correct=("faster_correct", "mean")).reset_index()
    g.to_csv(out / "by_quality_class.csv", index=False)
    audit.update({"kind": kind, "faster_correct": int(P.faster_correct.sum()), "n_points": len(P),
                  "mean_abs_pred_err_s": float((P.pred_diff_B_minus_A - P.meas_diff_B_minus_A).abs().mean()),
                  "quality_classes": P.quality_class.value_counts().to_dict()})
    if kind == "init":
        audit["router_choice_counts"] = P.router_choice.value_counts().to_dict()
        audit["router_A_and_only_B_passes"] = int(((P.router_choice == "A") & (P.quality_class == "only_B")).sum())
    json.dump(audit, open(out / "audit.json", "w"), indent=1)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(json.dumps(audit)); print(P.round(3).to_string()); print(g.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
