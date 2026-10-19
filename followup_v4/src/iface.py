"""followup_v4 interface variants (stage A). Environment semantics/answers/error handling are unchanged.

T0 = followup_v3 agent_tasks.TOOLS (unchanged).
T1 = same tools with explicit structure:
  - filter.conditions.items: object {column (string), op (enum of ops the env supports), value}, all required
  - aggregate.op: enum of ops the env supports; aggregate.by documented as optional
  - get_rows: start/n defaults documented (env defaults: start=0, n=5; n is capped by the env)
  - submit.answer: number, string or array (env accepts any JSON value)
R0 = src/mrp/router.py (unchanged).
R1 = tool-task router interface: same model descriptions (speed/quality claims as R0), roles USER/ASSISTANT/TOOL,
     the current user request always shown, latest tool result marked; same content token budget as R0
     (current request <= 1200 tokens, last 4 other messages <= 300 tokens each). Same output/parse/fallback.
"""
import copy
import re
import sys
import time
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
ROOT = V4.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "followup_v3/src"))
from mrp import router as R0  # noqa: E402
import agent_tasks as G  # noqa: E402

FILTER_OPS = ["==", "!=", "<", "<=", ">", ">=", "in"]
AGG_OPS = ["count", "sum", "mean", "median", "min", "max", "count_distinct"]

T0 = G.TOOLS
T1 = copy.deepcopy(G.TOOLS)
for t in T1:
    f = t["function"]
    if f["name"] == "filter":
        f["parameters"]["properties"]["conditions"]["items"] = {
            "type": "object",
            "properties": {"column": {"type": "string", "description": "exact column name from get_schema"},
                           "op": {"type": "string", "enum": FILTER_OPS},
                           "value": {"description": "number or string; a list of values when op is 'in'"}},
            "required": ["column", "op", "value"]}
    if f["name"] == "aggregate":
        f["parameters"]["properties"]["op"] = {"type": "string", "enum": AGG_OPS}
        f["parameters"]["properties"]["by"]["description"] = "optional column to group by"
    if f["name"] == "get_rows":
        f["parameters"]["properties"]["start"]["description"] = "first row index (default 0)"
        f["parameters"]["properties"]["n"]["description"] = "number of rows (default 5; the environment caps it)"
        f["parameters"]["properties"]["sort_by"]["description"] = "optional column to sort by"
    if f["name"] == "submit":
        f["parameters"]["properties"]["answer"] = {"description": "final answer: a number, a string or an array"}
TOOLSETS = {"T0": T0, "T1": T1}

# ----------------------------------------------------------------------------- R1
SYSTEM_R1 = ("You route each model call in a tool-using data-analysis session to one of two assistants that share the "
             "same conversation. The assistant answers user requests about a table by calling tools (get_schema, "
             "filter, aggregate, get_rows, submit). "
             "SMALL is Qwen3-1.7B: fast and cheap, fine for simple, single-operation requests. "
             "LARGE is Qwen3-8B: about 4 times slower, more reliable for multi-step logic, grouping, custom "
             "sorting and tie-breaking rules, or when previous attempts went wrong. "
             "Pick SMALL whenever SMALL is likely to handle the next step correctly; pick LARGE only when it is "
             "likely needed. Reply with exactly one word: SMALL or LARGE.")
ROLE = {"user": "USER", "assistant": "ASSISTANT", "tool": "TOOL"}
CUR_TOKENS, HIST_TOKENS, HIST_MESSAGES = R0.CUR_TOKENS, R0.HIST_TOKENS, R0.HIST_MESSAGES


def _trunc(tok, text, n):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= n else tok.decode(ids[:n]) + " …[truncated]"


def build_messages_r1(tok, conversation, active):
    msgs = [m for m in conversation if m["role"] != "system"]
    req_i = max(i for i, m in enumerate(msgs) if m["role"] == "user" and "[Request" in m["content"])
    request = msgs[req_i]["content"]
    others = [m for i, m in enumerate(msgs) if i != req_i and i > req_i][-HIST_MESSAGES:]
    lines = []
    last_tool = max((i for i, m in enumerate(others) if m["role"] == "tool"), default=None)
    for i, m in enumerate(others):
        tag = " [latest tool result]" if i == last_tool else ""
        lines.append(f"{ROLE.get(m['role'], m['role'].upper())}{tag}: {_trunc(tok, m['content'], HIST_TOKENS)}")
    hist = "\n\n".join(lines) if lines else "(no steps yet for this request)"
    user = (f"Current user request:\n{_trunc(tok, request, CUR_TOKENS)}\n\n"
            f"Most recent steps for this request (oldest first, truncated):\n{hist}\n\n"
            f"Currently active assistant: {R0.LABEL[active]}.\n"
            f"Which assistant should make the next model call? Reply SMALL or LARGE.")
    return [{"role": "system", "content": SYSTEM_R1}, {"role": "user", "content": user}]


def route(variant, server, conversation, active, max_tokens=4):
    if variant == "R0":
        return R0.route(server, conversation, "", active, max_tokens)
    t0 = time.perf_counter()
    msgs = build_messages_r1(server.tok, conversation, active)
    ids = server.render_ids(msgs)
    t_build = time.perf_counter() - t0
    call = server.complete(ids, max_tokens=max_tokens)
    choice = R0.parse(call["text"])
    return {"choice": choice if choice else R0.FALLBACK, "parse_error": choice is None, "raw": call["text"],
            "input_tokens": len(ids), "build_s": t_build,
            "call": {k: call[k] for k in ("ttft_s", "total_s", "completion_tokens", "cached_tokens", "error", "server")},
            "router_s": t_build + call["total_s"]}
