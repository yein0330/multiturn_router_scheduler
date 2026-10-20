"""Common loader for the offline analyses (no new inference).
Sources: followup_v3 dev50 (schema T0), followup_v4 stage A (T0/T1), followup_v4 stage B paired states."""
import collections, json, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
ROOT = V4.parent
sys.path.insert(0, str(ROOT / "followup_v3/src"))
from profile_v3 import CostModels  # noqa: E402

SESS = {json.loads(l)["session_id"]: json.loads(l) for l in open(ROOT / "followup_v3/data/dev.jsonl")}
CM = CostModels.load(ROOT / "followup_v3/results/stage01/profile/cost_models.json")


def _jl(p):
    return [json.loads(l) for l in open(p)]


def load_runs():
    """Return (sessions, calls) with uniform keys: session_id, context, schema, policy (+request/call for calls)."""
    S, C = [], []
    for s in _jl(ROOT / "followup_v3/results/runs/dev50/sessions.jsonl"):
        s = dict(s, schema="T0", policy={"L0": "L0_R0"}.get(s["policy"], s["policy"]), src="v3")
        S.append(s)
    for c in _jl(ROOT / "followup_v3/results/runs/dev50/calls.jsonl"):
        c = dict(c, schema="T0", policy={"L0": "L0_R0"}.get(c["policy"], c["policy"]), src="v3")
        C.append(c)
    for s in _jl(V4 / "results/stageA/sessions.jsonl"):
        S.append(dict(s, src="v4A"))
    for c in _jl(V4 / "results/stageA/calls.jsonl"):
        C.append(dict(c, src="v4A"))
    return S, C


def index(S, C):
    run = {(s["src"], s["session_id"], s["context"], s["schema"], s["policy"]): s for s in S}
    req = collections.defaultdict(list)
    for c in C:
        req[(c["src"], c["session_id"], c["context"], c["schema"], c["policy"], c["request"])].append(c)
    for v in req.values():
        v.sort(key=lambda c: c["call"])
    return run, req


def tool_sig(c):
    return [(t["name"], json.dumps(t.get("args"), sort_keys=True)) for t in c["tools"]]


def has_repeat(calls):
    """True if some tool call is identical (name+args) to the immediately preceding tool call in the request."""
    sig = [x for c in calls for x in tool_sig(c)]
    return any(a == b for a, b in zip(sig, sig[1:]))


def first_repeat_call_index(calls):
    prev = None
    for i, c in enumerate(calls):
        for x in tool_sig(c):
            if prev is not None and x == prev:
                return i
            prev = x
    return None


def prefill_c2(m, n, c):
    f = CM.fit[m]["C2"]  # [eta, alpha, beta, gamma, delta]; prefill part = eta + alpha n + beta n (c + n/2)
    return f[0] + f[1] * n + f[2] * n * (c + n / 2)


def difficulty(kind):
    return "easy" if kind.startswith("E") else ("hard" if kind.startswith("H") else "follow")


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
