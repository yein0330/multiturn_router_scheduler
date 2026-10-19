"""Stage B: same-state Follow/Deviate branches (single GPU owner; under run_locked.sh "A B R").

    python stageB.py select      (CPU; freezes configs/stageB_states.json from stage A T1/follow_R1 logs)
    python stageB.py check       (CPU; offline replay check of every stage-A T1/follow_R1 prompt hash)
    python stageB.py run         (GPU)
State = the moment before a model call in a stage-A T1/follow_R1 run. Restored: messages, tool env (handles),
current request / remaining call budget / previous request results, current model, and A/B/R prefix caches
(by replaying each server's earlier prompts). First call is forced (Follow = logged R1 recommendation,
Deviate = the other model); afterwards pi0 = follow_R1 with T1 (router re-applied to each branch's own history).
The branch-point router call is not repeated (its cost is common to both branches and excluded from both).
"""
import copy
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

V4 = Path(__file__).resolve().parents[1]
ROOT = V4.parent
sys.path.insert(0, str(V4 / "src"))
import runner_v4 as RV  # noqa: E402
import iface  # noqa: E402
from runner_v4 import G  # noqa: E402
from mrp.client import CacheTracker, Server  # noqa: E402

SESS = {json.loads(l)["session_id"]: json.loads(l) for l in open(ROOT / "followup_v3/data/dev.jsonl")}
CATS = ["first_call", "cur_A_rec_A", "cur_B_rec_B", "cur_A_rec_B", "cur_B_rec_A"]
BUDGET_AB, BUDGET_R = 8000, 4000


def h(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:16]


def source_runs():
    by = defaultdict(list)
    for l in open(V4 / "results/stageA/calls.jsonl"):
        c = json.loads(l)
        if c["schema"] == "T1" and c["policy"] == "follow_R1":
            by[(c["session_id"], c["context"])].append(c)
    for v in by.values():
        v.sort(key=lambda c: c["call"])
    return by


def category(c):
    rec = c["router"]["choice"]
    if c["active_before"] is None:
        return "first_call"
    return f"cur_{c['active_before']}_rec_{rec}"


def select():
    by = source_runs()
    cand = defaultdict(list)
    for (sid, ctx), cs in by.items():
        for c in cs:
            cand[(ctx, category(c))].append((sid, c["call"]))
    states, avail = [], {}
    for ctx in ("short", "long"):
        for cat in CATS:
            L = sorted(cand[(ctx, cat)], key=lambda x: hashlib.sha256(f"v4B|{x[0]}|{ctx}|{x[1]}".encode()).hexdigest())
            avail[f"{ctx}|{cat}"] = len(L)
            picked, used = [], set()
            for sid, call in L:  # distinct sessions first
                if len(picked) < 4 and sid not in used:
                    picked.append((sid, call)); used.add(sid)
            for sid, call in L:  # fill with repeats only if needed
                if len(picked) < 4 and (sid, call) not in picked:
                    picked.append((sid, call))
            for sid, call in picked:
                states.append({"state_id": f"{ctx}|{cat}|{sid}|{call}", "context": ctx, "category": cat,
                               "session_id": sid, "call": call})
    rng = random.Random(402)
    for s in states:
        s["order"] = rng.choice([["follow", "deviate"], ["deviate", "follow"]])
    rep = sorted(states, key=lambda s: hashlib.sha256(("v4Brep|" + s["state_id"]).encode()).hexdigest())[:10]
    for s in states:
        s["replicate"] = s in rep
    dep = defaultdict(int)
    for s in states:
        dep[(s["context"], s["session_id"])] += 1
    json.dump({"rule": "per context x category: candidates ordered by sha256('v4B|sid|ctx|call'); up to 4, distinct "
                       "sessions first; branch order per state random (seed 402); replicate = first 10 states by "
                       "sha256('v4Brep|state_id')", "available_candidates": avail, "states": states,
               "same_session_dependence": {f"{k[0]}|{k[1]}": v for k, v in dep.items() if v > 1}},
              open(V4 / "configs/stageB_states.json", "w"), indent=1)
    print(json.dumps(avail), len(states))


def env_hash(env):
    return h({k: [r[env.cols[0]] for r in v] for k, v in env.handles.items()})


def rebuild(srv_tok, rtok, sess, ctx, cs, upto):
    """Replay logged calls < upto through the runner's own message logic. Returns state + per-call prompts."""
    st = RV.new_state(sess, ctx)
    env, messages = st["env"], st["messages"]
    prompts, rprompts = {}, {}
    tok_srv = type("S", (), {"tok": srv_tok})()
    it = iter(cs)
    c = next(it)
    while st["k"] < len(sess["requests"]):
        k = st["k"]
        if not st["request_open"]:
            messages.append({"role": "user", "content": st["notice"] + G.request_message(sess, k)})
            st["notice"], st["n_in_req"], st["request_open"] = "", 0, True
            st["submitted"], st["answer"] = False, None
        while not st["submitted"] and st["n_in_req"] < G.MAX_CALLS_PER_REQUEST:
            ids = RV.render(tok_srv, messages, "T1")
            prompts[c["call"]] = ids
            if c["router"] is not None:
                rm = iface.build_messages_r1(rtok, messages, st["current"])
                rprompts[c["call"]] = rtok(rtok.apply_chat_template(rm, tokenize=False, add_generation_prompt=True),
                                           add_special_tokens=False)["input_ids"]
            if c["call"] == upto:
                return st, prompts, rprompts
            text = c["assistant_text"]
            messages.append({"role": "assistant", "content": text})
            tool_calls, perr = G.parse_tool_calls(text)
            for name, args in tool_calls[: G.MAX_TOOL_CALLS_PER_TURN]:
                res = env.call(name, args)
                messages.append({"role": "tool", "content": json.dumps(res)})
                if name == "submit" and "error" not in res:
                    st["submitted"], st["answer"] = True, args.get("answer") if isinstance(args, dict) else None
                    break
            if not tool_calls:
                messages.append({"role": "user", "content": "[Environment] No valid tool call found"
                                 + (f" ({perr[0]})" if perr else "") + ". Respond with a tool call."})
            st["current"] = c["chosen"]
            st["tracker"].add(c["chosen"], ids)
            st["call_idx"] += 1
            st["n_in_req"] += 1
            c = next(it, None)
        ok = st["submitted"] and G.correct(st["answer"], sess["expected"][k])
        st["req_results"].append({"request": k, "kind": sess["requests"][k]["kind"], "submitted": st["submitted"],
                                  "answer": st["answer"], "expected": sess["expected"][k], "correct": bool(ok),
                                  "calls": st["n_in_req"]})
        if not st["submitted"]:
            st["notice"] = "[Environment] The call budget for the previous request was exhausted.\n"
        st["request_open"] = False
        st["k"] += 1
    raise ValueError("call not found")


def tokenizers():
    cfg = json.load(open(ROOT / "configs/models.json"))["models"]
    from mrp.client import tokenizer
    return (tokenizer(cfg["A"]["hf_id"], cfg["A"]["revision"]),
            tokenizer(cfg["R"]["hf_id"], cfg["R"]["revision"], cfg["R"].get("local_path")))


def check():
    atok, rtok = tokenizers()
    by = source_runs()
    n = bad = rb = 0
    for (sid, ctx), cs in by.items():
        _, prompts, rprompts = rebuild(atok, rtok, SESS[sid], ctx, cs, cs[-1]["call"])
        for c in cs:
            n += 1
            bad += RV.sha(prompts[c["call"]]) != c["prompt_sha"]
            rb += c["router"] is not None and len(rprompts[c["call"]]) != c["router"]["input_tokens"]
    print("calls", n, "prompt sha mismatches", bad, "router length mismatches", rb)
    return bad == 0 and rb == 0


def run():
    cfg = json.load(open(ROOT / "configs/models.json"))
    http = httpx.Client()
    servers = {m: Server(m, cfg["models"][m], http) for m in ("A", "B", "R")}
    for s in servers.values():
        w = s.render_ids([{"role": "user", "content": "Write a Python function that adds two numbers."}])
        for _ in range(3):
            s.complete(w, max_tokens=32)
    plan = json.load(open(V4 / "configs/stageB_states.json"))["states"]
    by = source_runs()
    rd = V4 / "results/stageB"
    rd.mkdir(parents=True, exist_ok=True)
    out = {"run_id": "stageB", "fh": open(rd / "calls.jsonl", "a"), "sfh": open(rd / "sessions.jsonl", "a")}
    rfh = open(rd / "restore_audit.jsonl", "a")
    done = set()
    for l in open(rd / "sessions.jsonl"):
        j = json.loads(l)
        done.add((j["state_id"], j["branch"], j["rep"]))
    nab = sum(1 for l in open(V4 / "results/stageA/calls.jsonl"))
    nr = sum(1 for l in open(V4 / "results/stageA/calls.jsonl") if json.loads(l)["router"] is not None)
    for l in open(rd / "calls.jsonl"):
        j = json.loads(l)
        nab += 1
        nr += j["router"] is not None
    prep_ab = prep_r = 0
    units = [(s, b, 0) for s in plan for b in s["order"]] + [(s, b, 1) for s in plan if s["replicate"] for b in s["order"]]
    for i, (s, branch, rep) in enumerate(units):
        if (s["state_id"], branch, rep) in done:
            continue
        if nab >= BUDGET_AB - 40 or nr >= BUDGET_R - 40:
            print("BUDGET STOP before unit", i, nab, nr, flush=True)
            break
        cs = by[(s["session_id"], s["context"])]
        src = next(c for c in cs if c["call"] == s["call"])
        sess = SESS[s["session_id"]]
        st, prompts, rprompts = rebuild(servers["A"].tok, servers["R"].tok, sess, s["context"], cs, s["call"])
        ok_prompt = RV.sha(prompts[s["call"]]) == src["prompt_sha"]
        # restore caches: reset all, replay each server's earlier prompts (excluded from branch latency)
        for srv in servers.values():
            srv.reset_cache()
        time.sleep(0.2)
        t0 = time.perf_counter()
        for c in cs:
            if c["call"] >= s["call"]:
                break
            servers[c["chosen"]].complete(prompts[c["call"]], max_tokens=1); prep_ab += 1
            if c["router"] is not None:
                servers["R"].complete(rprompts[c["call"]], max_tokens=1); prep_r += 1
        prep_s = time.perf_counter() - t0
        time.sleep(0.3)
        rec_model = src["router"]["choice"]
        first = rec_model if branch == "follow" else ("A" if rec_model == "B" else "B")
        expected = st["tracker"].predicted_hit(first, prompts[s["call"]])
        ehash = env_hash(st["env"])
        budget_left = G.MAX_CALLS_PER_REQUEST - st["n_in_req"]
        tag = {"state_id": s["state_id"], "category": s["category"], "branch": branch, "rep": rep,
               "first_model": first, "rec_at_state": rec_model, "state_call": s["call"], "state_request": st["k"],
               "prev_requests_correct": [r["correct"] for r in st["req_results"]]}
        summ, calls = RV.run_episode(sess, "follow_R1", s["context"], "T1", servers, out, state=st, first_model=first, tag=tag)
        nab += len(calls)
        nr += sum(c["router"] is not None for c in calls)
        first_call = calls[0]
        rfh.write(json.dumps({"state_id": s["state_id"], "branch": branch, "rep": rep, "prompt_hash_ok": ok_prompt,
                              "tool_state_hash": ehash, "expected_first_hit": expected,
                              "actual_first_hit": first_call["cached_tokens"],
                              "first_hit_ok": expected == first_call["cached_tokens"], "prep_s": prep_s,
                              "remaining_call_budget_in_request": budget_left}) + "\n")
        rfh.flush()
        print(f"[{i+1}/{len(units)}] {s['state_id']} {branch} r{rep} first={first} q={summ['q']:.2f} S={summ['S']} "
              f"calls={summ['n_calls']} lat={summ['latency_s']:.1f}s ok={ok_prompt} hit={expected}/{first_call['cached_tokens']} "
              f"| AB={nab}(+prep {prep_ab}) R={nr}(+prep {prep_r})", flush=True)
    json.dump({"model_calls_total_incl_stageA": nab, "router_calls_total_incl_stageA": nr,
               "prep_model_calls": prep_ab, "prep_router_calls": prep_r}, open(rd / "budget.json", "w"), indent=1)


if __name__ == "__main__":
    {"select": select, "check": check, "run": run}[sys.argv[1]]()
