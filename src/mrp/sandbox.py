"""Run model-written `solve(rows)` in an isolated subprocess and compare with gold.

Used by both the hidden evaluator (post-hoc, never shown to policies, excluded from session time)
and the policy-visible verifier in S5 (example table only, included in session time).
"""
import json
import math
import re
import resource
import subprocess
import sys
import tempfile
import time

PY = "/usr/bin/python3.12"  # system interpreter, isolated mode; not the project venv
TIMEOUT_S = 5

RUNNER = r"""
import json, sys
src = open(sys.argv[1]).read()
tables = json.load(open(sys.argv[2]))
ns = {"__name__": "submission"}
out = []
try:
    exec(compile(src, "submission.py", "exec"), ns)
    fn = ns["solve"]
except Exception as e:
    print(json.dumps({"load_error": f"{type(e).__name__}: {e}"[:300]})); sys.exit(0)
for t in tables:
    try:
        r = fn([dict(x) for x in t])
        json.dumps(r)
        out.append({"ok": True, "value": r})
    except Exception as e:
        out.append({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
print(json.dumps({"results": out}))
"""

_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def extract_code(text):
    """Last fenced code block that defines solve(); None if absent."""
    blocks = [b for b in _BLOCK.findall(text or "") if re.search(r"def\s+solve\s*\(", b)]
    return blocks[-1] if blocks else None


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S))


def run(code, tables):
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/s.py", "w").write(code)
        open(f"{d}/t.json", "w").write(json.dumps(tables))
        open(f"{d}/r.py", "w").write(RUNNER)
        try:
            p = subprocess.run([PY, "-I", f"{d}/r.py", f"{d}/s.py", f"{d}/t.json"], cwd=d,
                               capture_output=True, text=True, timeout=TIMEOUT_S + 2, preexec_fn=_limits)
            res = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else \
                {"load_error": f"no output (rc={p.returncode}) {p.stderr[-200:]}"}
        except subprocess.TimeoutExpired:
            res = {"load_error": "Timeout"}
        except (json.JSONDecodeError, IndexError) as e:
            res = {"load_error": f"bad runner output: {e}"}
    res["exec_s"] = time.perf_counter() - t0
    return res


def norm(v):
    if isinstance(v, tuple):
        v = list(v)
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {str(k): norm(x) for k, x in v.items()}
    return v


def equal(a, b):
    a, b = norm(a), norm(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    return a == b


def evaluate(code, tables, golds):
    """Hidden evaluation: pass iff every table matches."""
    if code is None:
        return {"pass": False, "reason": "no_code", "exec_s": 0.0}
    r = run(code, tables)
    if "load_error" in r:
        return {"pass": False, "reason": "load_error", "detail": r["load_error"], "exec_s": r["exec_s"]}
    per = [x["ok"] and equal(x["value"], g) for x, g in zip(r["results"], golds)]
    return {"pass": all(per), "per_table": per, "exec_s": r["exec_s"],
            "reason": "ok" if all(per) else ("runtime_error" if not all(x["ok"] for x in r["results"])
                                              else "wrong_value")}


def verify(code, example_rows, gold_example):
    """Policy-visible verifier (S5): runs on the example table shown in the prompt."""
    if code is None:
        return {"status": "FORMAT_ERROR", "exec_s": 0.0,
                "message": "[Verifier] FORMAT ERROR: no ```python block defining solve(rows) was found."}
    r = run(code, [example_rows])
    if "load_error" in r:
        return {"status": "TOOL_ERROR", "exec_s": r["exec_s"],
                "message": f"[Verifier] ERROR while loading your code: {r['load_error']}"}
    x = r["results"][0]
    if not x["ok"]:
        return {"status": "TOOL_ERROR", "exec_s": r["exec_s"],
                "message": f"[Verifier] ERROR: solve raised {x['error']} on the first 12 rows shown above."}
    if equal(x["value"], gold_example):
        return {"status": "PASS", "exec_s": r["exec_s"],
                "message": "[Verifier] PASS on the first 12 rows shown above."}
    return {"status": "FAIL", "exec_s": r["exec_s"],
            "message": (f"[Verifier] FAIL on the first 12 rows shown above: solve returned "
                        f"{json.dumps(norm(x['value']))[:200]}, expected {json.dumps(norm(gold_example))[:200]}.")}
