"""Pre-run checks: (1) T1 schema vs environment on valid/invalid arguments; (2) R1 input budget and request
visibility on the v3 dev50 L0 conversations (offline, tokenizer only)."""
import collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import iface
from iface import G
from mrp.client import tokenizer
from mrp import router as R0
sys.path.insert(0, str(iface.ROOT / "followup_v3/src"))
from gate1 import rebuild_prompts

V3 = iface.ROOT / "followup_v3"
s = json.loads(open(V3 / "data/dev.jsonl").readline())
env = G.Env(s, "short")
col = env.cols
ok = []
for op in iface.FILTER_OPS:  # every op in the T1 enum must be accepted by the env
    v = [s["table"][0][col[1]]] if op == "in" else (5 if op in ("<", "<=", ">", ">=") else s["table"][0][col[1]])
    c = col[4] if op in ("<", "<=", ">", ">=") else col[1]
    ok.append((op, "error" not in env.call("filter", {"source": "table", "conditions": [{"column": c, "op": op, "value": v}]})))
for op in iface.AGG_OPS:
    ok.append(("agg:" + op, "error" not in env.call("aggregate", {"source": "table", "op": op, "column": col[4]})))
bad = [("missing column", {"source": "table", "conditions": [{"hotel": "==", "value": "x"}]}),
       ("op not in enum", {"source": "table", "conditions": [{"column": col[1], "op": "equals", "value": "x"}]}),
       ("missing value", {"source": "table", "conditions": [{"column": col[1], "op": "=="}]})]
errs = [(n, env.call("filter", a).get("error")) for n, a in bad]
print("valid accepted:", all(x for _, x in ok), ok)
print("invalid rejected:", errs)
# R1 budget/visibility on dev50 L0 conversations
rcfg = json.load(open(iface.ROOT / "configs/models.json"))["models"]["R"]
rtok = tokenizer(rcfg["hf_id"], rcfg["revision"], rcfg.get("local_path"))
acfg = json.load(open(iface.ROOT / "configs/models.json"))["models"]["A"]
asrv = type("S", (), {"tok": tokenizer(acfg["hf_id"], acfg["revision"])})()
sessions = {json.loads(l)["session_id"]: json.loads(l) for l in open(V3 / "data/dev.jsonl")}
C = [json.loads(l) for l in open(V3 / "results/runs/dev50/calls.jsonl")]
by = collections.defaultdict(list)
for c in C:
    if c["policy"] == "L0": by[(c["session_id"], c["context"])].append(c)
n = vis = 0; l0 = []; l1 = []
for (sid, ctx), cs in list(by.items()):
    cs = sorted(cs, key=lambda c: c["call"])
    for c in cs:
        msgs, _ = rebuild_prompts(asrv, sessions[sid], ctx, cs, c["call"])
        m1 = iface.build_messages_r1(rtok, msgs, c["active_before"])
        m0 = R0.build_messages(rtok, msgs, "", c["active_before"])
        t1 = len(rtok(rtok.apply_chat_template(m1, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"])
        t0 = len(rtok(rtok.apply_chat_template(m0, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"])
        l1.append(t1); l0.append(t0); n += 1
        req = [m for m in msgs if m["role"] == "user" and "[Request" in m["content"]][-1]["content"]
        vis += req[:200] in m1[1]["content"]
import numpy as np
print("R1 request visible", vis, "/", n)
print("router prompt tokens R0 p50/p95/max", np.percentile(l0, 50), np.percentile(l0, 95), max(l0),
      "| R1", np.percentile(l1, 50), np.percentile(l1, 95), max(l1))
json.dump({"T1_valid_accepted": ok, "T1_invalid_rejected": errs, "R1_request_visible": [vis, n],
           "router_tokens": {"R0": [float(np.percentile(l0, 50)), float(np.percentile(l0, 95)), int(max(l0))],
                             "R1": [float(np.percentile(l1, 50)), float(np.percentile(l1, 95)), int(max(l1))]}},
          open(iface.V4 / "results/precheck_iface.json", "w"), indent=1)
