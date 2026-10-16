"""Stage 1-2 (offline column): router score / rho for dev50 L0 calls (needs R server; under flock).
Added after the live run; NOT used online. Same method as stage 0-3 (full-candidate log-likelihood).
rho mapping = configs/rho_mapping.json (frozen in stage 0 from v2-task dev router calls). Because the task
changed, the empirical percentile of these new margins is also reported (dev50 itself), marked post hoc.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

V3 = Path(__file__).resolve().parents[1]
ROOT = V3.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(V3 / "src"))
from mrp import router  # noqa: E402
from mrp.client import Server  # noqa: E402
from stage0_router import score  # noqa: E402
from gate1 import rebuild_prompts, load_calls  # noqa: E402


def main():
    cfg = json.load(open(ROOT / "configs/models.json"))
    http = httpx.Client()
    R = Server("R", cfg["models"]["R"], http)
    A = Server("A", cfg["models"]["A"], http)  # tokenizer only (no request is sent to A)
    end = R.tok.convert_tokens_to_ids("<|im_end|>")
    cands = {"SMALL": R.tok("SMALL", add_special_tokens=False)["input_ids"] + [end],
             "LARGE": R.tok("LARGE", add_special_tokens=False)["input_ids"] + [end]}
    rho_map = np.array(json.load(open(V3 / "configs/rho_mapping.json"))["dev_margins_sorted"])
    sessions = {json.loads(l)["session_id"]: json.loads(l) for l in open(V3 / "data/dev.jsonl")}
    by = defaultdict(list)
    for c in load_calls("dev50"):
        if c["policy"] == "L0":
            by[(c["session_id"], c["context"])].append(c)
    out = open(V3 / "results/stage01/dev50_L0_router_scores.jsonl", "w")
    n = match_len = match_choice = 0
    for (sid, ctx), cs in by.items():
        cs = sorted(cs, key=lambda c: c["call"])
        for c in cs:
            msgs, _ = rebuild_prompts(A, sessions[sid], ctx, cs, c["call"])
            ids = R.render_ids(router.build_messages(R.tok, msgs, "", c["active_before"]))
            try:
                sc = score(R, ids, cands)
            except KeyError:
                body = {"model": R.name, "prompt": ids + cands["SMALL"], "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}
                err = R.http.post(f"{R.base}/v1/completions", json=body, timeout=120).text[:300]
                out.write(json.dumps({"session_id": sid, "context": ctx, "call": c["call"], "error": err,
                                      "router_prompt_tokens": len(ids)}) + "\n")
                n += 1
                continue
            m = sc["LARGE"] - sc["SMALL"]
            n += 1
            match_len += len(ids) == c["router"]["input_tokens"]
            match_choice += ("B" if m > 0 else "A") == c["router"]["choice"]
            out.write(json.dumps({"session_id": sid, "context": ctx, "call": c["call"], "margin": m,
                                  "rho_stage0_map": float(np.searchsorted(rho_map, m, side="right") / len(rho_map)),
                                  "logged_choice": c["router"]["choice"], "len_match": len(ids) == c["router"]["input_tokens"]}) + "\n")
    json.dump({"n": n, "len_match_rate": match_len / n, "score_argmax_match_rate": match_choice / n,
               "note": "offline scoring after the run; 2 extra prompt-only R calls per decision; not used online"},
              open(V3 / "results/stage01/dev50_router_score_audit.json", "w"), indent=1)
    print(n, match_len / n, match_choice / n)


if __name__ == "__main__":
    main()
