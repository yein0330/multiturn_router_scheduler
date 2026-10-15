"""Instrumented calls to the vLLM servers (pattern adapted from ~/switchcost/instrument.py).

- Prompts are rendered client-side and sent as token ids to /v1/completions.
- Per-request server metrics come from Prometheus counter/histogram deltas (concurrency 1 => the delta
  is this request). Missing values are None, never 0.
"""
import functools
import json
import re
import time

import httpx
from transformers import AutoTokenizer

BLOCK = 16
COUNTERS = ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total", "vllm:request_success_total", "vllm:num_preemptions_total"]
HISTS = ["vllm:request_prefill_time_seconds", "vllm:request_decode_time_seconds",
         "vllm:request_queue_time_seconds", "vllm:request_inference_time_seconds",
         "vllm:time_to_first_token_seconds"]
_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)$')


@functools.lru_cache(maxsize=4)
def tokenizer(hf_id, revision, local_path=None):
    return AutoTokenizer.from_pretrained(local_path or hf_id, revision=None if local_path else revision,
                                         local_files_only=True)


class Server:
    def __init__(self, role, mcfg, http: httpx.Client):
        self.role, self.cfg, self.http = role, mcfg, http
        self.base = f"http://127.0.0.1:{mcfg['port']}"
        self.name = mcfg.get("served_model_name", mcfg["hf_id"])
        self.tok = tokenizer(mcfg["hf_id"], mcfg["revision"], mcfg.get("local_path"))

    def render_ids(self, messages, enable_thinking=False):
        kw = {"enable_thinking": enable_thinking} if self.role in ("A", "B") else {}
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def scrape(self):
        out = {}
        for line in self.http.get(f"{self.base}/metrics", timeout=10).text.splitlines():
            if not line or line[0] == "#":
                continue
            m = _LINE.match(line)
            if m and not m.group(1).endswith("_bucket"):
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
        return out

    def reset_cache(self):
        t0 = time.perf_counter()
        r = self.http.post(f"{self.base}/reset_prefix_cache", timeout=30)
        return {"status": r.status_code, "s": time.perf_counter() - t0}

    def complete(self, prompt_ids, max_tokens, min_tokens=None, temperature=0.0):
        """Streaming completion; returns client timings + server metric deltas."""
        before = self.scrape()
        body = {"model": self.name, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": temperature,
                "stream": True, "stream_options": {"include_usage": True}}
        if min_tokens is not None:
            body.update({"min_tokens": min_tokens, "ignore_eos": True})
        rec = {"ttft_s": None, "total_s": None, "text": "", "finish_reason": None, "usage": None, "error": None}
        t0 = time.perf_counter()
        try:
            with self.http.stream("POST", f"{self.base}/v1/completions", json=body, timeout=600) as r:
                if r.status_code != 200:
                    rec["error"] = f"HTTP {r.status_code}: {r.read()[:300]!r}"
                else:
                    for line in r.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        j = json.loads(data)
                        if j.get("choices"):
                            c = j["choices"][0]
                            if rec["ttft_s"] is None and c.get("text"):
                                rec["ttft_s"] = time.perf_counter() - t0
                            rec["text"] += c.get("text") or ""
                            rec["finish_reason"] = c.get("finish_reason") or rec["finish_reason"]
                        if j.get("usage"):
                            rec["usage"] = j["usage"]
        except Exception as e:  # recorded, not swallowed
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["total_s"] = time.perf_counter() - t0
        # counters update asynchronously after the stream ends
        after, t1 = self.scrape(), time.time()
        while after.get("vllm:request_success_total", 0) - before.get("vllm:request_success_total", 0) < 1 \
                and time.time() - t1 < 5:
            time.sleep(0.03)
            after = self.scrape()
        keys = COUNTERS + [h + s for h in HISTS for s in ("_sum", "_count")]
        rec["server"] = {k: (after[k] - before[k]) if k in after and k in before else None for k in keys}
        rec["kv_usage_after"] = after.get("vllm:kv_cache_usage_perc")
        rec["prompt_tokens"] = len(prompt_ids)
        rec["completion_tokens"] = (rec["usage"] or {}).get("completion_tokens")
        rec["cached_tokens"] = rec["server"]["vllm:prefix_cache_hits_total"]
        return rec


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class CacheTracker:
    """Client-side view of what each server's prefix cache holds for *this* policy run.

    Only prompts are tracked: with Qwen3's template the next prompt diverges from the previous one at
    the assistant turn (the generation prompt contains an empty <think> block that past assistant
    turns do not), so generated-token blocks are never reusable. Predictions are checked against the
    server's prefix_cache_hits counter in every turn record.
    """

    def __init__(self):
        self.seqs = {}

    def predicted_hit(self, role, ids):
        best = max((lcp(ids, s) for s in self.seqs.get(role, [])), default=0)
        return min(best // BLOCK, (len(ids) - 1) // BLOCK) * BLOCK

    def add(self, role, ids):
        self.seqs.setdefault(role, []).append(list(ids))

    def evict(self, role):
        self.seqs[role] = []
