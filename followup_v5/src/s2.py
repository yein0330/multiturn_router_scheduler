"""Stage 2 (S2): request-start 'B needed' prediction.

    python s2.py states    (CPU) request-start states from always_A T1 paths (train 0-b, dev v4 stage A) + labels
    python s2.py score     (GPU R only, under run_locked.sh "R") P0 = R1 decision, P1 = R1s logprob score
    python s2.py analyze   (CPU) P0/P1/P2 metrics
Inputs available at request start only: current request text (incl. the interface header [New task]/[Follow-up]),
previous request's observable outcome on this path (submitted, budget exhausted, repeat occurred, tool errors).
Labels (analysis only): B needed = always_A wrong & always_B right on that request; A enough = always_A right;
both fail = neither. Hidden kind is never an input.
History used for the router at request start = the always_A path's history (a documented approximation).
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import V5, ROOT, G, iface, sha  # noqa: E402
import stageB as SB  # noqa: E402
import s1 as S1  # noqa: E402

OUT = V5 / "results/stage2"


def runs():
    R = {}
    for split, f in (("dev", ROOT / "followup_v4/results/stageA/sessions.jsonl"), ("train", V5 / "results/stage0/0b/sessions.jsonl")):
        for l in open(f):
            s = json.loads(l)
            if s["schema"] == "T1" and s["policy"] in ("always_A", "always_B"):
                R[(split, s["session_id"], s["context"], s["policy"])] = s
    return R


def states():
    OUT.mkdir(parents=True, exist_ok=True)
    R = runs()
    P = S1.paths()
    holdout = set()
    tr_ids = json.load(open(V5 / "configs/train_harvest_sessions.json"))["sessions"]
    for sc in G.PATTERNS:  # 20% of train sessions held out for dev (session level, by hash)
        c = sorted([x for x in tr_ids if f"-{sc}-" in x], key=lambda x: sha("v5s2hold|" + x))
        holdout |= set(c[:2])
    rows = []
    for (split, sid, ctx), cs in P.items():
        a, b = R[(split, sid, ctx, "always_A")], R[(split, sid, ctx, "always_B")]
        byreq = defaultdict(list)
        for c in cs:
            byreq[c["request"]].append(c)
        for k in sorted(byreq):
            first = byreq[k][0]
            prev = byreq.get(k - 1, [])
            ra, rb = a["requests"][k]["correct"], b["requests"][k]["correct"]
            sess = S1.DEV[sid] if split == "dev" else S1.TRAIN[sid]
            prev_texts = [c["assistant_text"] for c in prev]
            role = "dev" if split == "dev" else ("dev" if sid in holdout else "train")
            rows.append({"split": split, "role": role, "session_id": sid, "context": ctx, "request": k,
                         "state_call": first["call"], "text": G.request_message(sess, k),
                         "follow_up": "[Follow-up]" in G.request_message(sess, k),
                         "prev_submitted": a["requests"][k - 1]["submitted"] if k else None,
                         "prev_budget_exhausted": (not a["requests"][k - 1]["submitted"]) if k else False,
                         "prev_repeat": any(x == y for x, y in zip(prev_texts, prev_texts[1:])),
                         "prev_tool_errors": sum(1 for c in prev for t in c["tools"] if t["error"]),
                         "A_correct": ra, "B_correct": rb,
                         "label": "B_needed" if (not ra and rb) else ("A_enough" if ra else "both_fail"),
                         "kind": sess["requests"][k]["kind"]})  # kind: analysis only
    json.dump(rows, open(OUT / "states.json", "w"))
    c = Counter((r["role"], r["label"]) for r in rows)
    print(len(rows), dict(c), "holdout train sessions:", len(holdout))


R1S_SYSTEM = iface.SYSTEM_R1.replace("Reply with exactly one word: SMALL or LARGE.",
                                     "Reply with exactly one letter: A for SMALL or B for LARGE.")


def r1s_messages(tok, conversation, active):
    m = iface.build_messages_r1(tok, conversation, active)
    m[0]["content"] = R1S_SYSTEM
    m[1]["content"] = m[1]["content"].replace("Reply SMALL or LARGE.", "Reply A or B.")
    return m


def score():
    import httpx
    from mrp.client import Server
    cfg = json.load(open(ROOT / "configs/models.json"))
    http = httpx.Client()
    R = Server("R", cfg["models"]["R"], http)
    atok = SB.tokenizers()[0]
    rows = json.load(open(OUT / "states.json"))
    P = S1.paths()
    fh = open(OUT / "router_scores.jsonl", "w")
    n_missing = 0
    for i, r in enumerate(rows):
        sess = S1.DEV[r["session_id"]] if r["split"] == "dev" else S1.TRAIN[r["session_id"]]
        cs = P[(r["split"], r["session_id"], r["context"])]
        st, _, _ = SB.rebuild(atok, R.tok, sess, r["context"], cs, r["state_call"])
        msgs, cur = st["messages"], st["current"]
        R.reset_cache()
        rr = iface.route("R1", R, msgs, cur)
        ids = R.render_ids(r1s_messages(R.tok, msgs, cur))
        body = {"model": R.name, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "logprobs": 20}
        j = R.http.post(f"{R.base}/v1/completions", json=body, timeout=120).json()
        top = j["choices"][0]["logprobs"]["top_logprobs"][0]
        la, lb = top.get("A"), top.get("B")
        miss = la is None or lb is None
        n_missing += miss
        fh.write(json.dumps({"i": i, "R1_choice": rr["choice"], "R1_raw": rr["raw"], "R1s_text": j["choices"][0]["text"],
                             "lp_A": la, "lp_B": lb, "missing": miss,
                             "R1s_margin": None if miss else lb - la, "R1s_prompt_tokens": len(ids)}) + "\n")
        if i % 50 == 0:
            print(i, flush=True)
    print("done", len(rows), "missing candidates", n_missing)


# ----------------------------------------------------------------------------- analysis
TOK = re.compile(r"[a-z_]+|\d+")


def tfidf_fit(texts):
    docs = [TOK.findall(t.lower()) for t in texts]
    df = Counter(w for d in docs for w in set(d))
    vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= 2))}
    idf = np.zeros(len(vocab))
    for w, i in vocab.items():
        idf[i] = math.log((1 + len(docs)) / (1 + df[w])) + 1
    return vocab, idf


def tfidf(texts, vocab, idf):
    X = np.zeros((len(texts), len(vocab)))
    for r, t in enumerate(texts):
        for w in TOK.findall(t.lower()):
            if w in vocab:
                X[r, vocab[w]] += 1
    X = X * idf
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)


def obs(rows):
    return np.array([[float(r["follow_up"]), float(r["prev_budget_exhausted"]), float(r["prev_repeat"]),
                      min(r["prev_tool_errors"], 5) / 5, float(r["context"] == "long"), float(r["request"] == 0)] for r in rows])


def logreg(X, y, C, iters=300, lr=0.5):
    w = np.zeros(X.shape[1]); b = 0.0
    lam = 1.0 / C
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = X.T @ (p - y) / len(y) + lam * w / len(y)
        w -= lr * g * 10
        b -= lr * (p - y).mean() * 10
    return w, b


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def recall_at_precision(s, y, prec=0.6):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    best = (0.0, None)
    for t in np.unique(s):
        pred = s >= t
        if pred.sum() == 0:
            continue
        p = (pred & y).sum() / pred.sum()
        r = (pred & y).sum() / y.sum()
        if p >= prec and r > best[0]:
            best = (float(r), float(t))
    return best


def analyze():
    rows = json.load(open(OUT / "states.json"))
    sc = [json.loads(l) for l in open(OUT / "router_scores.jsonl")]
    for r, s in zip(rows, sc):
        r.update(s)
    out = {"counts": {}}
    for role in ("train", "dev"):
        out["counts"][role] = Counter(r["label"] for r in rows if r["role"] == role)
    allr = [r for r in rows]
    out["R1s_vs_R1_agreement"] = float(np.mean([("B" if (r["R1s_margin"] or 0) > 0 else "A") == r["R1_choice"]
                                                for r in allr if not r["missing"]]))
    out["R1s_missing"] = int(sum(r["missing"] for r in allr))
    res = []
    tr = [r for r in rows if r["role"] == "train"]
    dv = [r for r in rows if r["role"] == "dev"]
    for excl in (False, True):
        T = [r for r in tr if not (excl and r["label"] == "both_fail")]
        D = [r for r in dv if not (excl and r["label"] == "both_fail")]
        yT = np.array([r["label"] == "B_needed" for r in T], float)
        yD = np.array([r["label"] == "B_needed" for r in D], bool)
        tag = "exclude_both_fail" if excl else "both_fail_as_negative"
        # P0: R1 decision
        s0 = np.array([r["R1_choice"] == "B" for r in D], float)
        res.append({"setting": tag, "predictor": "P0_R1_decision", "n_dev": len(D), "pos_dev": int(yD.sum()),
                    "auc": auc(s0, yD), "recall": float((s0.astype(bool) & yD).sum() / yD.sum()) if yD.sum() else None,
                    "precision": float((s0.astype(bool) & yD).sum() / max(1, s0.sum()))})
        # P1: R1s margin
        D1 = [(r, y) for r, y in zip(D, yD) if not r["missing"]]
        s1 = np.array([r["R1s_margin"] for r, _ in D1]); y1 = np.array([y for _, y in D1])
        rp = recall_at_precision(s1, y1)
        res.append({"setting": tag, "predictor": "P1_R1s_margin", "n_dev": len(D1), "pos_dev": int(y1.sum()),
                    "auc": auc(s1, y1), "recall_at_prec0.6": rp[0], "threshold": rp[1]})
        # P2: TF-IDF + obs, logistic; C chosen on dev AUC
        vocab, idf = tfidf_fit([r["text"] for r in T])
        XT = np.hstack([tfidf([r["text"] for r in T], vocab, idf), obs(T)])
        XD = np.hstack([tfidf([r["text"] for r in D], vocab, idf), obs(D)])
        best = None
        for C in (0.1, 1.0, 10.0, 100.0):
            w, b = logreg(XT, yT, C)
            sD = XD @ w + b
            a = auc(sD, yD)
            if best is None or (a or 0) > best[0]:
                best = (a, C, sD)
        rp = recall_at_precision(best[2], yD)
        res.append({"setting": tag, "predictor": "P2_tfidf_obs_logreg", "n_train": len(T), "pos_train": int(yT.sum()),
                    "n_dev": len(D), "pos_dev": int(yD.sum()), "auc": best[0], "C": best[1],
                    "recall_at_prec0.6": rp[0], "threshold": rp[1], "vocab": len(vocab),
                    "note": "C selected on the same dev set (optimistic)"})
        if not excl:
            # confusion by hidden kind family (analysis only)
            fam = lambda k: "easy" if k.startswith("E") else ("hard" if k.startswith("H") else "follow")
            thr = rp[1] if rp[1] is not None else np.inf
            conf = defaultdict(lambda: Counter())
            for r, s, s0_ in zip(D, best[2], s0):
                conf[fam(r["kind"])][(r["label"], "P2:B" if s >= thr else "P2:A", "R1:" + ("B" if s0_ else "A"))] += 1
            out["confusion_by_kind_P2_P0_dev"] = {k: {"|".join(kk): v for kk, v in c.items()} for k, c in conf.items()}
    out["results"] = res
    out["phrasing_check"] = {"train_variant": sorted({r["text"][:0] or "variant0" for r in tr}),
                             "dev_note": "dev = v4 stage-A dev sessions (variant 1, domains hotel/gym) + held-out train sessions (variant 0)"}
    json.dump(out, open(OUT / "s2_results.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    {"states": states, "score": score, "analyze": analyze}[sys.argv[1]]()
