"""Stage 0-3: router score extraction (optional feature; design.md stage 0-3).

    python stage0_router.py      (needs the R server; run under flock via run_stage0_router.sh)
Score = conditional log-likelihood of the full candidate completion ("SMALL<|im_end|>" / "LARGE<|im_end|>")
after the exact router prompt, from vLLM prompt_logprobs (one extra prompt-only R call per candidate).
margin = ll(LARGE) - ll(SMALL). No forced/constrained decoding.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

V3 = Path(__file__).resolve().parents[1]
ROOT = V3.parent
sys.path.insert(0, str(ROOT / "src"))
from mrp import router, tasks  # noqa: E402
from mrp.client import Server  # noqa: E402

OUT0, OUT1 = V3 / "results/stage0", V3 / "results/stage01"
RUNS = {  # run dir -> session file
    "dev_smoke": "data/dev/smoke.jsonl", "dev_calib_sessions": "data/dev/calib_sessions.jsonl",
    "dev_calib_single": "data/dev/calib_single.jsonl", "dev_smoke_long_evict": "data/dev/smoke.jsonl",
    "evalA": "data/eval/eval.jsonl", "evalB_long_retain": "data/eval/eval.jsonl",
    "evalB_short_evict": "data/eval/eval.jsonl", "evalB_long_evict": "data/eval/eval.jsonl",
    "followup_v1/results/newpol": "data/eval/eval.jsonl"}


def h(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def conversation_at(sess, ts, turn, context):
    """Messages the policy had when deciding at `turn` (ending with that turn's user message)."""
    table = tasks.initial_user_text(sess, context)
    msgs = [{"role": "system", "content": tasks.SYSTEM}]
    for t in ts:
        st = sess["steps"][t["step"]]
        content = table + "\n" + st["request"] if (t["step"] == 0 and t["attempt"] == 0) else t["user_text"]
        msgs.append({"role": "user", "content": content})
        if t["turn"] == turn:
            return msgs, table
        msgs.append({"role": "assistant", "content": t["assistant_text"]})
    raise ValueError


def collect(R):
    items = []
    for run, sf in RUNS.items():
        d = ROOT / ("results/" + run if not run.startswith("followup") else run)
        sess = {json.loads(l)["session_id"]: json.loads(l) for l in open(ROOT / sf)}
        by = defaultdict(list)
        for l in open(d / "turns.jsonl"):
            t = json.loads(l)
            by[(t["session_id"], t["policy"], t["context"], t["cache_mode"])].append(t)
        for k, ts in by.items():
            ts.sort(key=lambda t: t["turn"])
            for t in ts:
                if not t["router"]:
                    continue
                msgs, table = conversation_at(sess[k[0]], ts, t["turn"], k[2])
                rm = router.build_messages(R.tok, msgs, table, t["active_before"])
                ids = R.render_ids(rm)
                items.append({"run": run, "split": "dev" if run.startswith("dev") else "eval", "session_id": k[0],
                              "policy": k[1], "context": k[2], "cache": k[3], "turn": t["turn"], "step": t["step"],
                              "attempt": t["attempt"], "active_before": t["active_before"],
                              "logged_input_tokens": t["router"]["input_tokens"], "logged_raw": t["router"]["raw"],
                              "logged_choice": t["router"]["choice"], "ids": ids, "ids_hash": h(ids),
                              "len_match": len(ids) == t["router"]["input_tokens"]})
    return items


def score(R, ids, cands):
    out = {}
    for name, cid in cands.items():
        body = {"model": R.name, "prompt": ids + cid, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}
        r = R.http.post(f"{R.base}/v1/completions", json=body, timeout=120).json()
        pl = r["choices"][0]["prompt_logprobs"]
        ll = 0.0
        for pos, tok_id in zip(range(len(ids), len(ids) + len(cid)), cid):
            ent = pl[pos]
            ll += ent[str(tok_id)]["logprob"]
        out[name] = ll
    return out


def main():
    OUT0.mkdir(parents=True, exist_ok=True); OUT1.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(ROOT / "configs/models.json"))
    R = Server("R", cfg["models"]["R"], httpx.Client())
    items = collect(R)
    end = R.tok.convert_tokens_to_ids("<|im_end|>")
    cands = {"SMALL": R.tok("SMALL", add_special_tokens=False)["input_ids"] + [end],
             "LARGE": R.tok("LARGE", add_special_tokens=False)["input_ids"] + [end]}
    uniq = {}
    for it in items:
        uniq.setdefault(it["ids_hash"], it["ids"])
    print("router calls", len(items), "unique prompts", len(uniq), flush=True)
    res = {}
    for i, (hh, ids) in enumerate(uniq.items()):
        g = R.complete(ids, max_tokens=4)
        sc = score(R, ids, cands)
        res[hh] = {"regen": g["text"], "ll_small": sc["SMALL"], "ll_large": sc["LARGE"]}
        if i % 200 == 0:
            print(i, flush=True)
    rows = []
    for it in items:
        r = res[it["ids_hash"]]
        m = r["ll_large"] - r["ll_small"]
        it2 = {k: v for k, v in it.items() if k != "ids"}
        it2.update({"regen": r["regen"], "regen_match": r["regen"] == it["logged_raw"],
                    "ll_small": r["ll_small"], "ll_large": r["ll_large"], "margin": m,
                    "score_choice": "B" if m > 0 else "A", "score_match": ("B" if m > 0 else "A") == it["logged_choice"]})
        rows.append(it2)
    with open(OUT0 / "router_margin.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    audit = {"candidate_tokenization": {k: v for k, v in cands.items()},
             "method": "full-candidate conditional log-likelihood via prompt_logprobs (2 extra prompt-only R calls per decision)",
             "first_token_logprob_used": False, "forced_decoding": False,
             "n_router_calls": n, "n_unique_prompts": len(uniq),
             "prompt_hash_logged": False, "len_match_rate": sum(r["len_match"] for r in rows) / n,
             "greedy_regen_match_rate": sum(r["regen_match"] for r in rows) / n,
             "score_argmax_match_rate": sum(r["score_match"] for r in rows) / n,
             "extra_overhead": "2 additional R prefill calls per decision; NOT an 'no extra LLM call' implementation"}
    dev_m = np.sort([r["margin"] for r in rows if r["split"] == "dev"])
    json.dump({"definition": "rho = empirical CDF of dev router margins (ll LARGE - ll SMALL); percentile, not a probability",
               "n_dev": len(dev_m), "dev_margins_sorted": dev_m.tolist()}, open(V3 / "configs/rho_mapping.json", "w"))
    json.dump(audit, open(OUT1 / "router_score_audit.json", "w"), indent=1)
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
