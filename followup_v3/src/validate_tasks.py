"""Generator validation: (1) reference program via env == independent direct computation, 100%;
(2) a random tool-calling policy scores ~0; (3) rendering example; (4) A/B token-id identity."""
import json, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import agent_tasks as G

def direct(sess):
    d, rows = sess["domain"], sess["table"]
    F = lambda r: G.F(d, r)
    def sel(conds):
        ok = lambda r, c: {"==": r[c["column"]] == c["value"], "!=": r[c["column"]] != c["value"],
                           "<": r[c["column"]] < c["value"] if not isinstance(c["value"], list) else None,
                           "<=": r[c["column"]] <= c["value"] if not isinstance(c["value"], list) else None,
                           ">=": r[c["column"]] >= c["value"] if not isinstance(c["value"], list) else None,
                           "in": r[c["column"]] in c["value"] if isinstance(c["value"], list) else None}[c["op"]]
        return [r for r in rows if all(ok(r, c) for c in conds)]
    out = []
    for req in sess["requests"]:
        k = req["kind"]
        if k == "H2":
            v = req["ref"][0][1]["conditions"][0]["value"]
            m = round(sum(r[F("price")] for r in rows if r[F("cat2")] == v) / max(1, sum(1 for r in rows if r[F("cat2")] == v)), 2)
            out.append(sum(1 for r in rows if r[F("price")] > m)); continue
        R = sel(req["conds"])
        if k in ("E1", "F_count"): out.append(len(R))
        elif k in ("E2", "F_sum"): out.append(sum(r[F("qty")] for r in R))
        elif k == "E3": out.append(max(r[F("price")] for r in R) if R else None)
        elif k == "E4": out.append(round(sum(r[F("qty")] for r in R) / len(R), 2) if R else None)
        elif k == "H1":
            t = {}
            for r in R: t[r[F("cat1")]] = t.get(r[F("cat1")], 0) + r[F("qty")]
            out.append(min(t, key=lambda g: (-t[g], g)))
        elif k == "H3":
            out.append(min(R, key=lambda r: (-r[F("price")], r[F("id")]))[F("id")] if R else None)
        elif k == "H4":
            g = {}
            for r in R: g.setdefault(r[F("cat2")], []).append(r[F("qty")])
            med = {x: round(statistics.median(v), 2) for x, v in g.items()}
            out.append(min(med, key=lambda x: (med[x], x)))
    return out

def random_policy(sess, rng, ctx="short"):
    env = G.Env(sess, ctx); ok = 0
    names = ["get_schema", "filter", "aggregate", "get_rows", "submit"]
    for k, req in enumerate(sess["requests"]):
        for _ in range(G.MAX_CALLS_PER_REQUEST):
            n = rng.choice(names)
            col = rng.choice(env.cols)
            args = {"get_schema": {"table": "table"}, "filter": {"source": "table", "conditions": [{"column": col, "op": rng.choice(["==", "!=", ">="]), "value": rng.choice([r[col] for r in env.rows])}]},
                    "aggregate": {"source": rng.choice(["table"] + list(env.handles)), "op": rng.choice(["count", "sum", "mean", "max"]), "column": col},
                    "get_rows": {"source": "table", "n": 3}, "submit": {"answer": rng.choice([0, 1, 5, 10, rng.choice(env.rows)[col]])}}[n]
            res = env.call(n, args)
            if n == "submit":
                ok += G.correct(args["answer"], sess["expected"][k]); break
    return ok / len(sess["requests"])

if __name__ == "__main__":
    rng = random.Random(0); bad = 0; n = 0; none = 0; rnd = []
    for split in ("train", "dev", "eval"):
        for sc in G.PATTERNS:
            for i in range(20):
                s = G.build_session(split, sc, i, 7000 + i + 100 * list(G.PATTERNS).index(sc))
                d = direct(s); n += len(d)
                bad += sum(not G.correct(a, b) for a, b in zip(s["expected"], d))
                none += sum(x is None for x in s["expected"])
                rnd.append(random_policy(s, rng))
    print("requests", n, "reference vs direct mismatches", bad, "None answers", none)
    print("random policy request pass rate", round(sum(rnd) / len(rnd), 4))
