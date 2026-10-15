"""Procedural multi-turn coding tasks over a small table.

Each step asks the assistant for the complete current version of `solve(rows)`.
- New task  ([New task] header): a fresh spec = row filters + one aggregation.
- Follow-up ([Follow-up] header): modifies the *current* spec. The assistant must edit its own
  previous code, so an early mistake propagates (error accumulation).

Difficulty is a property of the step's operation family (easy vs hard), fixed here and checked
against A/B pass rates on dev data only (see design.md). Dev and eval use disjoint domains
(field names / value pools), disjoint phrasing variants and disjoint seed ranges.

Gold answers are computed by the reference functions below on hidden tables that are never shown to
policies. The verifier (scenario S5 only) runs the submitted code on the shown example table.
"""
import json
import random
import statistics
from dataclasses import dataclass

# ----------------------------------------------------------------------------- domains
# role -> (field name, values or (lo, hi))
DOMAINS = {
    # dev
    "orders": {"split": "dev", "entity": "orders", "id": "order_id", "id_prefix": "ORD",
               "cat1": ("region", ["north", "south", "east", "west"]),
               "cat2": ("product", ["lamp", "desk", "chair", "shelf", "rug"]),
               "status": ("status", ["delivered", "delayed", "cancelled"]),
               "qty": ("quantity", (1, 20)), "price": ("unit_price", (5, 200)), "day": ("day", (1, 28))},
    "library": {"split": "dev", "entity": "library loans", "id": "loan_id", "id_prefix": "LN",
                "cat1": ("branch", ["central", "harbor", "hill", "park"]),
                "cat2": ("genre", ["mystery", "history", "poetry", "science", "travel"]),
                "status": ("state", ["returned", "overdue", "lost"]),
                "qty": ("copies", (1, 20)), "price": ("fee_cents", (5, 200)), "day": ("day", (1, 28))},
    # eval
    "clinic": {"split": "eval", "entity": "clinic visits", "id": "visit_id", "id_prefix": "VIS",
               "cat1": ("department", ["cardio", "derma", "neuro", "ortho"]),
               "cat2": ("doctor", ["kim", "lee", "park", "choi", "jung"]),
               "status": ("outcome", ["completed", "no_show", "rescheduled"]),
               "qty": ("duration_min", (1, 20)), "price": ("fee", (5, 200)), "day": ("day", (1, 28))},
    "fleet": {"split": "eval", "entity": "delivery trips", "id": "trip_id", "id_prefix": "TRP",
              "cat1": ("depot", ["alpha", "bravo", "charlie", "delta"]),
              "cat2": ("driver", ["ana", "ben", "cho", "dev", "eli"]),
              "status": ("result", ["on_time", "late", "aborted"]),
              "qty": ("packages", (1, 20)), "price": ("distance_km", (5, 200)), "day": ("day", (1, 28))},
    "warehouse": {"split": "eval", "entity": "shipments", "id": "shipment_id", "id_prefix": "SHP",
                  "cat1": ("site", ["busan", "incheon", "daegu", "gwangju"]),
                  "cat2": ("carrier", ["fastco", "shipit", "roadrun", "bluebox", "nextday"]),
                  "status": ("phase", ["shipped", "held", "returned"]),
                  "qty": ("units", (1, 20)), "price": ("weight_kg", (5, 200)), "day": ("day", (1, 28))},
}
STATUS_W = [0.5, 0.3, 0.2]
FIELD_ORDER = ["id", "cat1", "cat2", "status", "qty", "price", "day"]


def F(dom, role):
    return DOMAINS[dom][role][0] if role != "id" else DOMAINS[dom]["id"]


def gen_rows(dom, n, rng, ties=False):
    d = DOMAINS[dom]
    ids = rng.sample(range(1000, 10000), n)
    rows = []
    for i in ids:
        r = {d["id"]: f"{d['id_prefix']}-{i}",
             d["cat1"][0]: rng.choice(d["cat1"][1]),
             d["cat2"][0]: rng.choice(d["cat2"][1]),
             d["status"][0]: rng.choices(d["status"][1], STATUS_W)[0]}
        if ties:  # small value ranges -> many ties, exercises tie-break rules
            r[d["qty"][0]] = rng.randint(1, 3)
            r[d["price"][0]] = rng.choice([10, 20, 30, 40])
            r[d["day"][0]] = rng.randint(1, 6)
        else:
            r[d["qty"][0]] = rng.randint(*d["qty"][1])
            r[d["price"][0]] = rng.randint(*d["price"][1])
            r[d["day"][0]] = rng.randint(*d["day"][1])
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------- filters
# Each filter: role it touches, params sampler, predicate, text (two phrasing variants: 0=dev, 1=eval).
def _f_status_eq(dom, rng):
    return {"kind": "status_eq", "v": rng.choice(DOMAINS[dom]["status"][1][:2])}


def _f_status_neq(dom, rng):
    return {"kind": "status_neq", "v": rng.choice(DOMAINS[dom]["status"][1][1:])}


def _f_qty_ge(dom, rng):
    return {"kind": "qty_ge", "t": rng.randint(3, 8)}


def _f_day_le(dom, rng):
    return {"kind": "day_le", "t": rng.randint(15, 25)}


def _f_cat1_in(dom, rng):
    return {"kind": "cat1_in", "vals": sorted(rng.sample(DOMAINS[dom]["cat1"][1], 2))}


def _f_price_lt(dom, rng):
    return {"kind": "price_lt", "t": rng.randint(120, 180)}


FILTERS = {"status_eq": ("status", _f_status_eq), "status_neq": ("status", _f_status_neq),
           "qty_ge": ("qty", _f_qty_ge), "day_le": ("day", _f_day_le),
           "cat1_in": ("cat1", _f_cat1_in), "price_lt": ("price", _f_price_lt)}


def pred(dom, f):
    k = f["kind"]
    if k == "status_eq":
        return lambda r: r[F(dom, "status")] == f["v"]
    if k == "status_neq":
        return lambda r: r[F(dom, "status")] != f["v"]
    if k == "qty_ge":
        return lambda r: r[F(dom, "qty")] >= f["t"]
    if k == "day_le":
        return lambda r: r[F(dom, "day")] <= f["t"]
    if k == "cat1_in":
        return lambda r: r[F(dom, "cat1")] in f["vals"]
    if k == "price_lt":
        return lambda r: r[F(dom, "price")] < f["t"]
    raise KeyError(k)


def filter_text(dom, f, variant):
    k = f["kind"]
    s, q, d, c, p = (F(dom, x) for x in ("status", "qty", "day", "cat1", "price"))
    body = {
        "status_eq": [f"`{s}` is '{f.get('v')}'", f"`{s}` equals '{f.get('v')}'"],
        "status_neq": [f"`{s}` is not '{f.get('v')}'", f"`{s}` differs from '{f.get('v')}'"],
        "qty_ge": [f"`{q}` is at least {f.get('t')}", f"`{q}` is {f.get('t')} or more"],
        "day_le": [f"`{d}` is at most {f.get('t')}", f"`{d}` is {f.get('t')} or less"],
        "cat1_in": [f"`{c}` is '{f.get('vals', ['', ''])[0]}' or '{f.get('vals', ['', ''])[1]}'",
                    f"`{c}` is one of '{f.get('vals', ['', ''])[0]}', '{f.get('vals', ['', ''])[1]}'"],
        "price_lt": [f"`{p}` is below {f.get('t')}", f"`{p}` is less than {f.get('t')}"],
    }[k]
    return body[variant]


# ----------------------------------------------------------------------------- aggregations
@dataclass(frozen=True)
class Agg:
    name: str
    difficulty: str  # "easy" | "hard"


def _sorted_by(rows, dom, *keys):
    return sorted(rows, key=lambda r: tuple(r[F(dom, k)] for k in keys))


def agg_gold(dom, a, rows):
    """Reference implementation. `a` is the agg dict (name + params)."""
    n = a["name"]
    I, c1, c2, s, q, p, d = (F(dom, x) for x in FIELD_ORDER)
    if n == "count":
        return len(rows)
    if n == "sum_qty":
        return sum(r[q] for r in rows)
    if n == "max_price_id":
        if not rows:
            return None
        return min(rows, key=lambda r: (-r[p], r[I]))[I]
    if n == "distinct_cat2":
        return sorted({r[c2] for r in rows})
    if n == "mean_qty":
        return round(sum(r[q] for r in rows) / len(rows), 2) if rows else 0.0
    if n == "group_argmax_rev":
        if not rows:
            return None
        tot = {}
        for r in rows:
            tot[r[c1]] = tot.get(r[c1], 0) + r[q] * r[p]
        return min(tot, key=lambda g: (-tot[g], g))
    if n == "median_by_cat2":
        g = {}
        for r in rows:
            g.setdefault(r[c2], []).append(r[q])
        return {k: round(statistics.median(v), 1) for k, v in g.items()}
    if n == "top3_multikey":
        return [r[I] for r in sorted(rows, key=lambda r: (-r[p], r[d], r[I]))[:3]]
    if n == "rate_threshold":
        g = {}
        for r in rows:
            t = g.setdefault(r[c1], [0, 0])
            t[0] += r[s] == a["v"]
            t[1] += 1
        pct = {k: round(100 * h / m, 1) for k, (h, m) in g.items()}
        return [[k, v] for k, v in sorted(pct.items(), key=lambda kv: (-kv[1], kv[0])) if v >= a["p"]]
    if n == "running_balance":
        bal, best, best_day = 0, None, None
        for r in sorted(rows, key=lambda r: (r[d], r[I])):
            if r[s] == a["s1"]:
                bal += r[q]
            elif r[s] == a["s2"]:
                bal -= r[q]
            else:
                continue
            if best is None or bal < best:
                best, best_day = bal, r[d]
        return [0, None] if best is None else [best, best_day]
    if n == "longest_day_streak":
        days = sorted({r[d] for r in rows})
        if not days:
            return [0, None]
        best_len, best_start, cur_len, cur_start = 1, days[0], 1, days[0]
        for prev, x in zip(days, days[1:]):
            if x == prev + 1:
                cur_len += 1
            else:
                cur_len, cur_start = 1, x
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        return [best_len, best_start]
    if n == "second_cat2":
        tot = {}
        for r in rows:
            tot[r[c2]] = tot.get(r[c2], 0) + r[q]
        order = sorted(tot, key=lambda g: (-tot[g], g))
        return order[1] if len(order) >= 2 else None
    raise KeyError(n)


EASY = ["count", "sum_qty", "max_price_id", "distinct_cat2", "mean_qty"]
HARD = ["group_argmax_rev", "median_by_cat2", "top3_multikey", "rate_threshold",
        "running_balance", "longest_day_streak", "second_cat2"]


def sample_agg(dom, name, rng):
    a = {"name": name}
    st = DOMAINS[dom]["status"][1]
    if name == "rate_threshold":
        a["v"] = rng.choice(st)
        a["p"] = rng.choice([20, 25, 30, 35])
    if name == "running_balance":
        a["s1"], a["s2"] = rng.sample(st, 2)
    return a


def agg_text(dom, a, variant):
    n = a["name"]
    I, c1, c2, s, q, p, d = (F(dom, x) for x in FIELD_ORDER)
    T = {
        "count": ["returns the number of rows (an int)",
                  "returns how many rows there are, as an int"],
        "sum_qty": [f"returns the sum of `{q}` over the rows (an int; 0 if there are no rows)",
                    f"returns the total of the `{q}` values as an int (0 when there are no rows)"],
        "max_price_id": [f"returns the `{I}` of the row with the largest `{p}`; if several rows tie, return the "
                         f"smallest such `{I}` (string comparison); return None if there are no rows",
                         f"returns the `{I}` whose `{p}` is highest, breaking ties by the smallest `{I}` "
                         f"as a string, or None when there are no rows"],
        "distinct_cat2": [f"returns a sorted list of the distinct `{c2}` values",
                          f"returns the distinct values of `{c2}` as an ascending sorted list"],
        "mean_qty": [f"returns the mean of `{q}` rounded with round(x, 2), or 0.0 if there are no rows",
                     f"returns the average `{q}` as round(average, 2), returning 0.0 for no rows"],
        "group_argmax_rev": [
            f"groups the rows by `{c1}`, computes for each group the sum of `{q}` * `{p}`, and returns the `{c1}` "
            f"value with the largest sum (if tied, the alphabetically smallest one); None if there are no rows",
            f"returns the `{c1}` whose total of `{q}` times `{p}` (summed over its rows) is the highest, choosing "
            f"the alphabetically first `{c1}` on ties, and None when there are no rows"],
        "median_by_cat2": [
            f"returns a dict mapping each `{c2}` value that occurs to the median of its `{q}` values (for an even "
            f"count, the mean of the two middle values), rounded with round(x, 1)",
            f"builds a dict from every `{c2}` present to the median `{q}` of its rows (two middle values averaged "
            f"when the count is even), each rounded via round(x, 1)"],
        "top3_multikey": [
            f"returns the list of `{I}` values of the first 3 rows when the rows are ordered by `{p}` descending, "
            f"then `{d}` ascending, then `{I}` ascending (fewer than 3 if there are fewer rows)",
            f"sorts rows by `{p}` from high to low, ties by `{d}` from low to high, remaining ties by `{I}` "
            f"ascending, and returns the `{I}` values of the top 3 rows as a list (shorter if fewer rows)"],
        "rate_threshold": [
            f"computes, for each `{c1}` value, the percentage of its rows whose `{s}` is '{a.get('v')}' as "
            f"round(100 * count / group_size, 1), and returns a list of [value, percentage] pairs for the groups "
            f"whose percentage is at least {a.get('p')}, sorted by percentage descending and then value ascending",
            f"for every `{c1}`, takes round(100 * (rows with `{s}` == '{a.get('v')}') / (rows in the group), 1) as "
            f"its percentage; returns [value, percentage] pairs only for percentages >= {a.get('p')}, ordered by "
            f"percentage high to low with ties by value alphabetically"],
        "running_balance": [
            f"walks through the rows sorted by (`{d}`, `{I}`) ascending with a balance starting at 0: a row whose "
            f"`{s}` is '{a.get('s1')}' adds its `{q}`, a row whose `{s}` is '{a.get('s2')}' subtracts its `{q}`, "
            f"and other rows are skipped. After each non-skipped row, record the balance. Return [lowest recorded "
            f"balance, `{d}` of the first row at which that lowest balance was recorded], or [0, None] if no row "
            f"was recorded",
            f"processes rows in ascending (`{d}`, `{I}`) order keeping a running balance from 0; '{a.get('s1')}' "
            f"rows (by `{s}`) add `{q}`, '{a.get('s2')}' rows subtract `{q}`, all others are ignored and not "
            f"recorded. Returns [the minimum balance seen right after a non-ignored row, the `{d}` of the earliest "
            f"such row reaching it], or [0, None] when nothing was recorded"],
        "longest_day_streak": [
            f"takes the set of distinct `{d}` values among the rows and returns [length, first_day] of the longest "
            f"run of consecutive integers in that set (if tied, the earliest run); [0, None] if there are no rows",
            f"finds, among the distinct `{d}` numbers present, the longest block of consecutive integers and "
            f"returns [its length, its starting day], preferring the earliest block on ties, or [0, None] for "
            f"no rows"],
        "second_cat2": [
            f"computes the total `{q}` for each `{c2}`, orders the `{c2}` values by total descending (ties "
            f"alphabetically), and returns the second one in that order; None if fewer than two `{c2}` values occur",
            f"sums `{q}` per `{c2}`, ranks the `{c2}` values from highest total to lowest (equal totals in "
            f"alphabetical order), and returns the one ranked second, or None when fewer than two are present"],
    }
    return T[n][variant]


# ----------------------------------------------------------------------------- spec / steps
def spec_gold(dom, spec, rows):
    keep = [r for r in rows if all(pred(dom, f)(r) for f in spec["filters"])]
    return agg_gold(dom, spec["agg"], keep)


def _new_filter(dom, spec, rng):
    used = {FILTERS[f["kind"]][0] for f in spec["filters"]}
    kinds = [k for k, (role, _) in FILTERS.items() if role not in used]
    k = rng.choice(kinds)
    return FILTERS[k][1](dom, rng)


def make_step(dom, kind, difficulty, prev_spec, rng, variant):
    """Return (spec_after_step, request_text, step_meta)."""
    if kind == "new":
        spec = {"filters": [], "agg": None}
        spec["filters"].append(_new_filter(dom, spec, rng))
        pool = EASY if difficulty == "easy" else HARD
        spec["agg"] = sample_agg(dom, rng.choice(pool), rng)
        ftxt = filter_text(dom, spec["filters"][0], variant)
        if variant == 0:
            text = (f"[New task] Write a new function `solve(rows)` that only considers rows whose {ftxt}, "
                    f"and {agg_text(dom, spec['agg'], variant)}.")
        else:
            text = (f"[New task] Please implement a fresh `solve(rows)`. Keep only rows where {ftxt}. "
                    f"Then it {agg_text(dom, spec['agg'], variant)}.")
        op = spec["agg"]["name"]
    else:
        spec = json.loads(json.dumps(prev_spec))
        if difficulty == "easy":  # add one more row filter; aggregation unchanged
            f = _new_filter(dom, spec, rng)
            spec["filters"].append(f)
            ftxt = filter_text(dom, f, variant)
            text = (f"[Follow-up] Update your current `solve(rows)`: in addition to the existing conditions, it "
                    f"must also only consider rows whose {ftxt}. Keep everything else the same."
                    if variant == 0 else
                    f"[Follow-up] Change the current `solve(rows)` so that, on top of the conditions it already "
                    f"applies, only rows where {ftxt} are used. Nothing else changes.")
            op = "add_filter:" + f["kind"]
        else:  # replace the aggregation by a different hard one; filters kept
            cur = spec["agg"]["name"]
            spec["agg"] = sample_agg(dom, rng.choice([h for h in HARD if h != cur]), rng)
            text = (f"[Follow-up] Update your current `solve(rows)`: keep exactly the same row conditions, but "
                    f"instead of what it returns now, it {agg_text(dom, spec['agg'], variant)}."
                    if variant == 0 else
                    f"[Follow-up] Rework the current `solve(rows)`: the row conditions stay as they are, but "
                    f"rather than its present result it now {agg_text(dom, spec['agg'], variant)}.")
            op = "swap_hard:" + spec["agg"]["name"]
    return spec, text, {"kind": kind, "difficulty": difficulty, "op": op}


# ----------------------------------------------------------------------------- scenarios
# Fixed before any eval run. Sessions in a scenario alternate sub-types by index (5/5 for 10 sessions).
# Structure letters: N = new task (exposed boundary), F = follow-up (no boundary marker).
SCENARIOS = {
    "S1_constant": [("EEEEE", "NFFNF"), ("HHHHH", "NFFNF")],
    "S2_shift": [("EEHHH", None), ("HHEEE", None)],          # structure set per index below
    "S3_alternating": [("EHEHE", "NFNFN"), ("HEHEH", "NFNFN")],
    "S4_accumulation": [("HEEEE", "NFFFF"), ("EHEEE", "NFFFF")],
    "S5_verifier": [("EEHHH", None), ("EHEHE", "NFNFN")],
}
SHIFT_STRUCT = {0: "NFNFF", 1: "NFFNF"}  # change at step 3: 0 -> exposed boundary, 1 -> inside a task


def session_layout(scenario, idx):
    """Sub-type and structure for session `idx` (0-based) in a scenario."""
    sub = idx % 2
    pattern, struct = SCENARIOS[scenario][sub]
    if struct is None:  # S2 both subtypes and S5 subtype 0: alternate boundary/within by (idx // 2)
        struct = SHIFT_STRUCT[(idx // 2) % 2]
    return sub, pattern, struct


SYSTEM = ("You are a careful Python programmer helping a user analyse a table. Answer every request with the "
          "complete current version of the function `solve(rows)` in a single ```python code block and nothing "
          "else. `rows` is a list of dicts with the fields described by the user. Use only the Python standard "
          "library. Do not print, read files, or call input().")


def data_block(dom, rows):
    d = DOMAINS[dom]
    fields = ", ".join(f"`{F(dom, r)}`" for r in FIELD_ORDER)
    lines = "\n".join(json.dumps(r) for r in rows)
    return (f"I have a table of {d['entity']}. Each row is a dict with the fields {fields}. "
            f"Here are {len(rows)} rows from the table, one JSON object per line:\n{lines}\n")


def build_session(split, scenario, idx, seed, long_rows=220, short_rows=12):
    rng = random.Random(seed)
    doms = [k for k, v in DOMAINS.items() if v["split"] == split]
    dom = doms[(seed + idx) % len(doms)]
    variant = 0 if split == "dev" else 1
    sub, pattern, struct = session_layout(scenario, idx)
    example = gen_rows(dom, short_rows, rng)
    extra = gen_rows(dom, long_rows - short_rows, rng)
    # ids must stay unique across example + extra
    seen = {r[F(dom, "id")] for r in example}
    extra = [r for r in extra if r[F(dom, "id")] not in seen]
    hidden = [gen_rows(dom, 40, rng), gen_rows(dom, 60, rng), gen_rows(dom, 30, rng, ties=True)]
    steps, spec = [], None
    for k, (dl, sl) in enumerate(zip(pattern, struct)):
        diff = "easy" if dl == "E" else "hard"
        kind = "new" if sl == "N" else "follow"
        spec, text, meta = make_step(dom, kind, diff, spec, rng, variant)
        gold = [spec_gold(dom, spec, t) for t in hidden]
        gold_example = spec_gold(dom, spec, example)
        steps.append({"step": k, "request": text, "spec": spec, "gold_hidden": gold,
                      "gold_example": gold_example, **meta})
    return {"session_id": f"{split}-{scenario}-{idx:02d}", "split": split, "scenario": scenario,
            "index": idx, "seed": seed, "domain": dom, "phrasing_variant": variant,
            "subtype": sub, "pattern": pattern, "structure": struct,
            "verifier": scenario == "S5_verifier",
            "example_rows": example, "extra_rows": extra, "hidden_tables": hidden, "steps": steps}


def initial_user_text(sess, context):
    rows = sess["example_rows"] if context == "short" else sess["example_rows"] + sess["extra_rows"]
    return data_block(sess["domain"], rows)
