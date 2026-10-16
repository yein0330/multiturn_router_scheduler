"""followup_v3 tool-using table-query agent tasks: environment, generator, reference solver, checker.

Protocol: Qwen3 native tool calling (<tool_call>{json}</tool_call>), tool results as role "tool".
Fixed rules (design.md stage 1; instruction 10-4):
- Each session = one backing table (rows never shown in full) + 4 requests. Handles are session-scoped
  and immutable; "table" always refers to the original table, so recovery is always possible.
- Tools: get_schema, filter, aggregate, get_rows, submit. Protocol errors (unknown tool/column/handle/op,
  bad JSON, missing args, type mismatch) always return {"error": ...} in every condition.
- Error exposure (session attribute): exposed sessions additionally return warnings that are observable
  from tool state only (filter value never occurs in the column; aggregate over 0 rows). Hidden sessions
  return the same data without warnings. 0 rows is a normal result, not an error.
- submit(answer) ends the current request regardless of correctness; correctness is never revealed.
- Budgets: max 8 model calls per request; max 3 tool calls executed per model turn (extra ignored, noted);
  a turn without a parsable tool call gets an environment notice and consumes one call.
  Budget exhausted -> request scored as failed, next request starts.
- Expected answers are computed by a reference program on the session's own backing table.
  Numbers: rounded to 2 decimals, compared with abs tol 0.01; strings: case-insensitive, trimmed;
  lists: exact order after normalization.
"""
import copy
import hashlib
import json
import random
import re
import statistics

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mrp import tasks as T  # noqa: E402

MAX_CALLS_PER_REQUEST = 8
MAX_TOOL_CALLS_PER_TURN = 3

DOMAINS = dict(T.DOMAINS)
DOMAINS.update({  # dev-only domains for v3
    "hotel": {"split": "dev", "entity": "hotel bookings", "id": "booking_id", "id_prefix": "BKG",
              "cat1": ("hotel", ["seaside", "airport", "downtown", "garden"]),
              "cat2": ("room_type", ["single", "double", "suite", "family", "studio"]),
              "status": ("state", ["checked_out", "cancelled", "no_show"]),
              "qty": ("nights", (1, 20)), "price": ("rate", (5, 200)), "day": ("day", (1, 28))},
    "gym": {"split": "dev", "entity": "gym sessions", "id": "session_id", "id_prefix": "GYM",
            "cat1": ("branch", ["north", "south", "east", "west"]),
            "cat2": ("class_type", ["yoga", "spin", "boxing", "pilates", "hiit"]),
            "status": ("attendance", ["attended", "late", "missed"]),
            "qty": ("minutes", (1, 20)), "price": ("fee", (5, 200)), "day": ("day", (1, 28))},
})
SPLIT_DOMAINS = {"train": ["orders", "library"], "dev": ["hotel", "gym"], "eval": ["clinic", "fleet", "warehouse"]}
VARIANT = {"train": 0, "dev": 1, "eval": 2}
COL_DESC = {"id": "unique row identifier", "cat1": "location/group", "cat2": "category",
            "status": "outcome status", "qty": "count-like amount", "price": "numeric value", "day": "day of month"}

TOOLS = [
    {"type": "function", "function": {"name": "get_schema", "description": "Describe the columns of a table or handle, with sample rows.",
                                      "parameters": {"type": "object", "properties": {"table": {"type": "string", "description": "'table' or a handle id"}}, "required": ["table"]}}},
    {"type": "function", "function": {"name": "filter", "description": "Keep rows of a source (the original 'table' or a handle) that satisfy ALL conditions. Returns a new handle.",
                                      "parameters": {"type": "object", "properties": {
                                          "source": {"type": "string", "description": "'table' or a handle id such as 'h2'"},
                                          "conditions": {"type": "array", "description": "list of {column, op, value}; op one of ==, !=, <, <=, >, >=, in (value is a list for in)",
                                                         "items": {"type": "object"}}}, "required": ["source", "conditions"]}}},
    {"type": "function", "function": {"name": "aggregate", "description": "Aggregate a column of a source. op: count, sum, mean, median, min, max, count_distinct. Optional by=column groups the result.",
                                      "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "op": {"type": "string"},
                                                                                      "column": {"type": "string"}, "by": {"type": "string"}}, "required": ["source", "op", "column"]}}},
    {"type": "function", "function": {"name": "get_rows", "description": "Return rows of a source, optionally sorted (ties broken by id ascending).",
                                      "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "start": {"type": "integer"}, "n": {"type": "integer"},
                                                                                      "sort_by": {"type": "string"}, "descending": {"type": "boolean"}}, "required": ["source"]}}},
    {"type": "function", "function": {"name": "submit", "description": "Submit the final answer for the current request (a number, a string or a list). Ends the request.",
                                      "parameters": {"type": "object", "properties": {"answer": {}}, "required": ["answer"]}}},
]

SYSTEM = ("You are a data analysis agent. Answer each user request about the table named 'table' by calling tools. "
          "Use exact column names from get_schema. Call exactly the tools you need, one step at a time, and finish "
          "every request by calling submit with only the final answer (numbers rounded to 2 decimals when not "
          "integers). Do not explain; respond only with tool calls.")

CONTEXT = {"short": {"schema_samples": 3, "rows_cap": 10, "filter_preview": 0},
           "long": {"schema_samples": 40, "rows_cap": 40, "filter_preview": 15}}


def F(dom, role):
    return DOMAINS[dom]["id"] if role == "id" else DOMAINS[dom][role][0]


def gen_rows(dom, n, rng):
    """Copy of mrp.tasks.gen_rows (non-tie branch) using this module's DOMAINS (adds dev domains)."""
    d = DOMAINS[dom]
    ids = rng.sample(range(1000, 10000), n)
    rows = []
    for i in ids:
        rows.append({d["id"]: f"{d['id_prefix']}-{i}", d["cat1"][0]: rng.choice(d["cat1"][1]),
                     d["cat2"][0]: rng.choice(d["cat2"][1]), d["status"][0]: rng.choices(d["status"][1], T.STATUS_W)[0],
                     d["qty"][0]: rng.randint(*d["qty"][1]), d["price"][0]: rng.randint(*d["price"][1]),
                     d["day"][0]: rng.randint(*d["day"][1])})
    return rows


def r2(x):
    return round(float(x), 2)


# ----------------------------------------------------------------------------- environment
class Env:
    def __init__(self, sess, context):
        self.s = sess
        self.dom = sess["domain"]
        self.rows = sess["table"]
        self.cols = [F(self.dom, r) for r in T.FIELD_ORDER]
        self.numeric = {F(self.dom, r) for r in ("qty", "price", "day")}
        self.handles = {}
        self.k = 0
        self.ctx = CONTEXT[context]
        self.exposed = sess["exposed"]

    def _src(self, name):
        if name == "table":
            return self.rows
        if name in self.handles:
            return self.handles[name]
        raise KeyError(f"unknown source '{name}' (use 'table' or an existing handle)")

    def _check_col(self, c):
        if c not in self.cols:
            raise KeyError(f"unknown column '{c}'")

    def call(self, name, args):
        try:
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
            fn = {"get_schema": self.get_schema, "filter": self.filter, "aggregate": self.aggregate,
                  "get_rows": self.get_rows, "submit": self.submit}.get(name)
            if fn is None:
                raise KeyError(f"unknown tool '{name}'")
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments: {str(e)[:120]}"}
        except (KeyError, ValueError) as e:
            return {"error": str(e).strip("'\"")[:200]}

    def get_schema(self, table="table"):
        src = self._src(table)
        desc = {F(self.dom, r): {"type": "number" if F(self.dom, r) in self.numeric else "string", "meaning": COL_DESC[r]}
                for r in T.FIELD_ORDER}
        return {"source": table, "rows": len(src), "columns": desc, "sample_rows": src[: self.ctx["schema_samples"]]}

    def _pred(self, cond):
        c, op, v = cond.get("column"), cond.get("op"), cond.get("value")
        self._check_col(c)
        ops = {"==": lambda x: x == v, "!=": lambda x: x != v, "<": lambda x: x < v, "<=": lambda x: x <= v,
               ">": lambda x: x > v, ">=": lambda x: x >= v, "in": lambda x: x in v}
        if op not in ops:
            raise ValueError(f"unknown op '{op}'")
        if op == "in" and not isinstance(v, list):
            raise ValueError("value for 'in' must be a list")
        if op in ("<", "<=", ">", ">=") and (c not in self.numeric or not isinstance(v, (int, float))):
            raise ValueError(f"op '{op}' needs a numeric column and numeric value")
        return lambda r: ops[op](r[c])

    def filter(self, source, conditions):
        src = self._src(source)
        if not isinstance(conditions, list):
            raise ValueError("conditions must be a list")
        preds = [self._pred(c) for c in conditions]
        out = [r for r in src if all(p(r) for p in preds)]
        self.k += 1
        h = f"h{self.k}"
        self.handles[h] = out
        res = {"handle": h, "rows": len(out)}
        if self.ctx["filter_preview"]:
            res["preview"] = out[: self.ctx["filter_preview"]]
        if self.exposed:
            warn = []
            for c in conditions:
                col, op, v = c.get("column"), c.get("op"), c.get("value")
                vals = {r[col] for r in self.rows}
                if op in ("==", "!=") and v not in vals:
                    warn.append(f"value {json.dumps(v)} never occurs in column '{col}'")
                if op == "in":
                    miss = [x for x in v if x not in vals]
                    if miss:
                        warn.append(f"values {json.dumps(miss)} never occur in column '{col}'")
            if warn:
                res["warning"] = "; ".join(warn)
        return res

    def aggregate(self, source, op, column, by=None):
        src = self._src(source)
        self._check_col(column)
        if by is not None:
            self._check_col(by)
        if op not in ("count", "sum", "mean", "median", "min", "max", "count_distinct"):
            raise ValueError(f"unknown op '{op}'")
        if op in ("sum", "mean", "median", "min", "max") and column not in self.numeric:
            raise ValueError(f"op '{op}' needs a numeric column")

        def agg(rows):
            v = [r[column] for r in rows]
            if op == "count":
                return len(v)
            if op == "count_distinct":
                return len(set(v))
            if not v:
                return None
            return {"sum": lambda: sum(v), "mean": lambda: r2(sum(v) / len(v)), "median": lambda: r2(statistics.median(v)),
                    "min": lambda: min(v), "max": lambda: max(v)}[op]()
        if by is None:
            res = {"result": agg(src)}
        else:
            groups = {}
            for r in src:
                groups.setdefault(r[by], []).append(r)
            res = {"result": {k: agg(g) for k, g in sorted(groups.items())}}
        if self.exposed and len(src) == 0:
            res["warning"] = "aggregate over 0 rows"
        return res

    def get_rows(self, source, start=0, n=5, sort_by=None, descending=False):
        src = self._src(source)
        if not isinstance(start, int) or not isinstance(n, int) or start < 0 or n < 0:
            raise ValueError("start and n must be non-negative integers")
        n = min(n, self.ctx["rows_cap"])
        rows = sorted(src, key=lambda r: r[self.cols[0]])  # id ascending
        if sort_by is not None:
            self._check_col(sort_by)
            rows = sorted(rows, key=lambda r: r[sort_by], reverse=bool(descending))  # stable: ties keep id order
        return {"rows": rows[start:start + n], "total": len(src), "n_cap": self.ctx["rows_cap"]}

    def submit(self, answer=None):
        return {"submitted": True}


# ----------------------------------------------------------------------------- requests (reference programs)
def _cond(dom, role, op, v):
    return {"column": F(dom, role), "op": op, "value": v}


def gen_request(dom, kind, rng, prev_conds, variant):
    """Return dict(text, kind, difficulty, conds (gold selection), solve(env)->answer, ref_calls)."""
    st, c1, c2 = DOMAINS[dom]["status"][1], DOMAINS[dom]["cat1"][1], DOMAINS[dom]["cat2"][1]
    q, p, d, s, a, b, I = (F(dom, r) for r in ("qty", "price", "day", "status", "cat1", "cat2", "id"))
    W = lambda txt: txt[variant] if isinstance(txt, list) else txt
    if kind == "E1":
        v = rng.choice(c2)
        conds = [_cond(dom, "cat2", "==", v)]
        text = W([f"How many rows have {b} '{v}'?", f"Count the rows whose {b} is '{v}'.", f"What is the number of rows with {b} equal to '{v}'?"])
        ref = [("filter", {"source": "table", "conditions": conds}), ("aggregate", {"source": "$h", "op": "count", "column": I})]
    elif kind == "E2":
        v = rng.choice(st)
        conds = [_cond(dom, "status", "==", v)]
        text = W([f"What is the total {q} over rows whose {s} is '{v}'?", f"Sum {q} for rows with {s} '{v}'.", f"Give the sum of {q} where {s} equals '{v}'."])
        ref = [("filter", {"source": "table", "conditions": conds}), ("aggregate", {"source": "$h", "op": "sum", "column": q})]
    elif kind == "E3":
        v = rng.choice(c1)
        conds = [_cond(dom, "cat1", "==", v)]
        text = W([f"What is the highest {p} among rows with {a} '{v}'?", f"Find the maximum {p} for {a} '{v}'.", f"Report the largest {p} in rows where {a} is '{v}'."])
        ref = [("filter", {"source": "table", "conditions": conds}), ("aggregate", {"source": "$h", "op": "max", "column": p})]
    elif kind == "E4":
        t = rng.randint(10, 20)
        conds = [_cond(dom, "day", "<=", t)]
        text = W([f"What is the average {q} over rows with {d} at most {t}? Round to 2 decimals.",
                  f"Compute the mean {q} for rows where {d} <= {t} (2 decimals).", f"Give the mean {q} among rows with {d} no greater than {t}, rounded to 2 decimals."])
        ref = [("filter", {"source": "table", "conditions": conds}), ("aggregate", {"source": "$h", "op": "mean", "column": q})]
    elif kind == "H1":
        v = rng.choice(st)
        conds = [_cond(dom, "status", "!=", v)]
        text = W([f"Excluding rows whose {s} is '{v}', which {a} has the largest total {q}? If tied, the alphabetically first. Answer with the {a} value.",
                  f"Ignore rows with {s} '{v}'. Return the {a} whose summed {q} is highest (ties: alphabetical first).",
                  f"Among rows not having {s} '{v}', give the {a} with the greatest total {q}; break ties alphabetically."])
        ref = [("filter", {"source": "table", "conditions": conds}), ("aggregate", {"source": "$h", "op": "sum", "column": q, "by": a}), ("argmax_alpha", None)]
    elif kind == "H2":
        v = rng.choice(c2)
        conds = None  # two-stage: threshold computed from the table
        text = W([f"How many rows have a {p} strictly greater than the average {p} of rows with {b} '{v}'?",
                  f"Count rows whose {p} exceeds the mean {p} of the '{v}' {b} rows.",
                  f"Take the average {p} over rows with {b} '{v}'; how many rows (in the whole table) have {p} above it?"])
        ref = [("filter", {"source": "table", "conditions": [_cond(dom, "cat2", "==", v)]}),
               ("aggregate", {"source": "$h", "op": "mean", "column": p}), ("count_above", p)]
    elif kind == "H3":
        vs = sorted(rng.sample(c1, 2))
        t = rng.randint(5, 12)
        conds = [_cond(dom, "cat1", "in", vs), _cond(dom, "qty", ">=", t)]
        text = W([f"Among rows with {a} in {vs} and {q} at least {t}, which {I} has the highest {p}? If tied, the smallest {I}.",
                  f"For {a} '{vs[0]}' or '{vs[1]}' with {q} >= {t}, give the {I} of the row with maximum {p} (ties: smallest {I}).",
                  f"Restrict to {a} in {vs} and {q} of {t} or more; report the {I} whose {p} is largest, smallest {I} on ties."])
        ref = [("filter", {"source": "table", "conditions": conds}), ("get_rows", {"source": "$h", "start": 0, "n": 1, "sort_by": p, "descending": True}), ("first_id", I)]
    elif kind == "H4":
        v = rng.choice(st)
        conds = [_cond(dom, "status", "==", v)]
        text = W([f"For rows with {s} '{v}', which {b} has the smallest median {q}? If tied, the alphabetically first.",
                  f"Restricting to {s} '{v}', return the {b} with the lowest median {q} (ties: alphabetical first).",
                  f"Among rows where {s} is '{v}', give the {b} whose median {q} is smallest; alphabetical first on ties."])
        ref = [("filter", {"source": "table", "conditions": conds}), ("aggregate", {"source": "$h", "op": "median", "column": q, "by": b}), ("argmin_alpha", None)]
    elif kind in ("F_count", "F_sum"):
        used = {c["column"] for c in prev_conds}
        opts = [r for r in ("status", "day", "price", "cat2", "cat1") if F(dom, r) not in used]
        role = rng.choice(opts)
        if role == "status":
            nc = _cond(dom, "status", "!=", rng.choice(st))
            phr = f"{s} is not '{nc['value']}'"
        elif role == "day":
            nc = _cond(dom, "day", ">=", rng.randint(5, 12))
            phr = f"{d} is at least {nc['value']}"
        elif role == "price":
            nc = _cond(dom, "price", "<", rng.randint(120, 170))
            phr = f"{p} is below {nc['value']}"
        elif role == "cat2":
            nc = _cond(dom, "cat2", "!=", rng.choice(c2))
            phr = f"{b} is not '{nc['value']}'"
        else:
            nc = _cond(dom, "cat1", "!=", rng.choice(c1))
            phr = f"{a} is not '{nc['value']}'"
        conds = prev_conds + [nc]
        what = "how many rows remain" if kind == "F_count" else f"what is the total {q} of those rows"
        text = W([f"Keep the rows you selected for the previous request and additionally require that {phr}. Now {what}?",
                  f"Starting from your previous selection, also require {phr}; {what}?",
                  f"Continue from the previous request's rows, add the condition that {phr}, and tell me {what}."])
        ref = [("filter", {"source": "$prev", "conditions": [nc]}),
               ("aggregate", {"source": "$h", "op": "count" if kind == "F_count" else "sum", "column": I if kind == "F_count" else q})]
    else:
        raise KeyError(kind)
    return {"kind": kind, "difficulty": "hard" if kind.startswith("H") else ("follow" if kind.startswith("F") else "easy"),
            "text": text, "conds": conds, "ref": ref}


def run_reference(env, req, prev_handle):
    """Execute the reference program through the real tool env; returns (answer, handle_for_followups)."""
    h, last = None, None
    for step, args in req["ref"]:
        if step in ("filter", "aggregate", "get_rows"):
            a = json.loads(json.dumps(args).replace('"$h"', json.dumps(h)).replace('"$prev"', json.dumps(prev_handle)))
            last = env.call(step, a)
            assert "error" not in last, (step, a, last)
            if step == "filter":
                h = last["handle"]
        elif step == "argmax_alpha":
            d = last["result"]
            return min(d, key=lambda k: (-d[k], k)), h
        elif step == "argmin_alpha":
            d = last["result"]
            return min(d, key=lambda k: (d[k], k)), h
        elif step == "count_above":
            thr = last["result"]
            last = env.call("filter", {"source": "table", "conditions": [{"column": args, "op": ">", "value": thr}]})
            return last["rows"], h
        elif step == "first_id":
            return (last["rows"][0][args] if last["rows"] else None), h
    return last["result"], h


def expected_answers(sess):
    env = Env(sess, "short")
    out, prev = [], None
    for req in sess["requests"]:
        ans, h = run_reference(env, req, prev)
        prev = h if req["conds"] is not None else prev
        out.append(ans)
    return out


# ----------------------------------------------------------------------------- sessions
PATTERNS = {  # scenario -> list of (subtype pattern); E=easy new, H=hard new, F=follow-up on previous selection
    "const": [["E", "E", "E", "E"], ["H", "H", "H", "H"]],
    "rise": [["E", "E", "H", "H"], ["E", "H", "H", "H"]],
    "alt": [["E", "H", "E", "H"], ["H", "E", "H", "E"]],
    "chain": [["H3", "F", "F", "F"], ["E", "F", "F", "F"]],
    "expose": [["E", "H", "F", "H"], ["H", "F", "E", "H"]],
}


def build_session(split, scenario, idx, seed, n_rows=80):
    rng = random.Random(seed)
    doms = SPLIT_DOMAINS[split]
    dom = doms[(seed + idx) % len(doms)]
    variant = VARIANT[split]
    pat = PATTERNS[scenario][idx % 2]
    table = gen_rows(dom, n_rows, rng)
    reqs, prev_conds = [], None
    for k, sym in enumerate(pat):
        if sym == "E":
            kind = rng.choice(["E1", "E2", "E3", "E4"])
        elif sym == "H":
            kind = rng.choice(["H1", "H2", "H3", "H4"])
        elif sym == "H3":
            kind = "H3"
        else:
            kind = rng.choice(["F_count", "F_sum"])
            if prev_conds is None:  # follow-up needs a selection; fall back to easy new task
                kind = rng.choice(["E1", "E2", "E3", "E4"])
        req = gen_request(dom, kind, rng, prev_conds or [], variant)
        req["boundary"] = "Follow-up" if kind.startswith("F") else "New task"
        reqs.append(req)
        if req["conds"] is not None:
            prev_conds = req["conds"]
    sess = {"session_id": f"{split}-{scenario}-{idx:02d}", "split": split, "scenario": scenario, "index": idx,
            "seed": seed, "domain": dom, "variant": variant, "pattern": "".join(r["kind"] for r in reqs),
            "exposed": scenario == "expose", "table": table, "requests": reqs}
    sess["expected"] = expected_answers(sess)
    return sess


def request_message(sess, k):
    r = sess["requests"][k]
    return f"[Request {k + 1}/{len(sess['requests'])}] [{r['boundary']}] {r['text']}"


# ----------------------------------------------------------------------------- parsing and scoring
_TC = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_tool_calls(text):
    calls, errors = [], []
    for raw in _TC.findall(text or ""):
        try:
            j = json.loads(raw)
            if not isinstance(j, dict) or "name" not in j:
                raise ValueError("missing name")
            calls.append((j["name"], j.get("arguments", {})))
        except Exception as e:  # noqa: BLE001 - recorded as a protocol error
            errors.append(f"invalid tool call JSON: {str(e)[:80]}")
    return calls, errors


def norm_answer(x):
    if isinstance(x, str):
        s = x.strip().strip("'\"").strip()
        try:
            return float(s) if re.fullmatch(r"-?\d+(\.\d+)?", s) else s.lower()
        except ValueError:
            return s.lower()
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, list) and len(x) == 1:
        return norm_answer(x[0])
    if isinstance(x, dict) and "answer" in x:
        return norm_answer(x["answer"])
    return x


def correct(submitted, expected):
    a, b = norm_answer(submitted), norm_answer(expected)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 0.01
    return a == b


def sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()
