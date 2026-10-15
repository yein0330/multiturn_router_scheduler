"""Fixed zero-shot LLM router R (exploratory baseline; NOT a trained or validated router).

Input = what the policy has observed so far: the last two prior exchanges (each message truncated to
its first HIST_TOKENS tokens, table rows omitted) + the current user message in full (table rows
omitted) + which assistant is currently active. No gold, no future steps, no difficulty labels.
Output = one label (SMALL/LARGE). No probability is requested or used.
Parse rule: first occurrence of SMALL or LARGE (case-insensitive) in the output; otherwise parse_error
and the fallback label is LARGE (fixed before evaluation).
"""
import re
import time

HIST_MESSAGES = 4      # last two prior exchanges
HIST_TOKENS = 300      # head-truncation per prior message
CUR_TOKENS = 1200      # head-truncation of the current message (table rows are already removed)
FALLBACK = "B"

SYSTEM = ("You route each user request to one of two coding assistants that share the same conversation. "
          "SMALL is Qwen3-1.7B: fast and cheap, fine for simple, single-operation requests. "
          "LARGE is Qwen3-8B: about 4 times slower, more reliable for multi-step logic, grouping, custom "
          "sorting and tie-breaking rules, or when previous attempts went wrong. "
          "Pick SMALL whenever SMALL is likely to answer correctly; pick LARGE only when it is likely needed. "
          "Reply with exactly one word: SMALL or LARGE.")

LABEL = {"A": "SMALL", "B": "LARGE", None: "none yet"}


def strip_table(text, table_text):
    return text.replace(table_text, "[table rows omitted]\n") if table_text else text


def _trunc(tok, text, n):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= n else tok.decode(ids[:n]) + " …[truncated]"


def build_messages(tok, conversation, table_text, active):
    """conversation: the policy's own message list ending with the current user message."""
    prior = [m for m in conversation[:-1] if m["role"] != "system"][-HIST_MESSAGES:]
    lines = []
    for m in prior:
        who = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"{who}: {_trunc(tok, strip_table(m['content'], table_text), HIST_TOKENS)}")
    cur = _trunc(tok, strip_table(conversation[-1]["content"], table_text), CUR_TOKENS)
    hist = "\n\n".join(lines) if lines else "(no earlier messages)"
    user = (f"Earlier conversation (oldest first, truncated):\n{hist}\n\n"
            f"Current request:\n{cur}\n\n"
            f"Currently active assistant: {LABEL[active]}.\n"
            f"Which assistant should answer the current request? Reply SMALL or LARGE.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(text):
    m = re.search(r"\b(SMALL|LARGE)\b", text or "", re.I)
    if not m:
        return None
    return "A" if m.group(1).upper() == "SMALL" else "B"


def route(server, conversation, table_text, active, max_tokens):
    t0 = time.perf_counter()
    msgs = build_messages(server.tok, conversation, table_text, active)
    ids = server.render_ids(msgs)
    t_build = time.perf_counter() - t0
    call = server.complete(ids, max_tokens=max_tokens)
    choice = parse(call["text"])
    return {"choice": choice if choice else FALLBACK, "parse_error": choice is None,
            "raw": call["text"], "input_tokens": len(ids), "build_s": t_build,
            "call": {k: call[k] for k in ("ttft_s", "total_s", "completion_tokens", "cached_tokens",
                                          "error", "server")},
            "router_s": t_build + call["total_s"]}
