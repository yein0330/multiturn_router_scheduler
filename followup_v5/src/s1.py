"""Stage 1 (S1): repeat-triggered escalation x handed-over history, same-state branches.

    python s1.py candidates      (CPU) list first-repeat states from always_A T1 paths (dev = v4 stage A, train = 0-b)
    python s1.py select          (CPU) freeze configs/s1_states.json (hash order, distinct sessions first)
    python s1.py check           (CPU) offline replay check of the selected states (prompt hashes)
    python s1.py run [--smoke N] (GPU, under run_locked.sh "A B R")
State = in an always_A path, right after the tool results of the first call whose assistant text equals the
previous call's text within the same request, before the next model call (the logged path must continue the
request; otherwise excluded and counted).
Branches (forced model for the rest of the current request, then pi0 = follow_R1 on T1 over each branch's history):
  a0 A continue | a1 B, original history | a2 B, repeats removed | a3 B, programmatic state summary | a4 A + repeat notice
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import V5, ROOT, G, iface, sha  # noqa: E402
import runner_v4 as RV  # noqa: E402
import stageB as SB  # noqa: E402  (followup_v4 replay)
from mrp.client import CacheTracker, Server  # noqa: E402

BRANCHES = ["a0", "a1", "a2", "a3", "a4"]
BRANCH_MODEL = {"a0": "A", "a1": "B", "a2": "B", "a3": "B", "a4": "A"}
NOTICE = ("\nNote: you repeated the previous call exactly. The result will not change. "
          "Use a different approach or submit.")
DEV = {json.loads(l)["session_id"]: json.loads(l) for l in open(ROOT / "followup_v3/data/dev.jsonl")}
TRAIN = {json.loads(l)["session_id"]: json.loads(l) for l in open(V5 / "data/train_pool.jsonl")}


def paths():
    """always_A T1 paths: dev from v4 stage A, train from v5 0-b."""
    out = {}
    for split, f in (("dev", ROOT / "followup_v4/results/stageA/calls.jsonl"), ("train", V5 / "results/stage0/0b/calls.jsonl")):
        by = defaultdict(list)
        for l in open(f):
            c = json.loads(l)
            if c["schema"] == "T1" and c["policy"] == "always_A":
                by[(c["session_id"], c["context"])].append(c)
        for k, v in by.items():
            out[(split,) + k] = sorted(v, key=lambda c: c["call"])
    return out


def candidates(write=True):
    P = paths()
    cands, excl = [], defaultdict(int)
    for (split, sid, ctx), cs in P.items():
        byreq = defaultdict(list)
        for c in cs:
            byreq[c["request"]].append(c)
        for r, rc in byreq.items():
            for i in range(1, len(rc)):
                if rc[i]["assistant_text"] == rc[i - 1]["assistant_text"]:
                    if i + 1 < len(rc):
                        cands.append({"split": split, "session_id": sid, "context": ctx, "request": r,
                                      "repeat_call": rc[i]["call"], "state_call": rc[i + 1]["call"]})
                    else:
                        excl[f"{split}|{ctx}|request ended at the repeat call"] += 1
                    break
    if write:
        json.dump({"n": len(cands), "excluded": excl, "candidates": cands}, open(V5 / "results/stage1/candidates.json", "w"), indent=1)
    return cands, excl


def select(max_states=80):
    cands, excl = candidates()
    states = []
    for split in ("dev", "train"):
        for ctx in ("short", "long"):
            L = [c for c in cands if c["split"] == split and c["context"] == ctx]
            L.sort(key=lambda c: sha(f"v5s1|{c['session_id']}|{ctx}|{c['state_call']}"))
            picked, used = [], set()
            for c in L:
                if len(picked) < max_states // 4 and c["session_id"] not in used:
                    picked.append(c); used.add(c["session_id"])
            for c in L:
                if len(picked) < max_states // 4 and c not in picked:
                    picked.append(c)
            states += picked
    rng = random.Random(504)
    for s in states:
        s["state_id"] = f"{s['split']}|{s['context']}|{s['session_id']}|{s['state_call']}"
        order = BRANCHES[:]
        rng.shuffle(order)
        s["order"] = order
    rep = sorted(states, key=lambda s: sha("v5s1rep|" + s["state_id"]))[:15]
    for s in states:
        s["replicate"] = s in rep
    avail = defaultdict(int)
    for c in cands:
        avail[f"{c['split']}|{c['context']}"] += 1
    json.dump({"rule": "first-repeat states from always_A T1 paths; per (split, context) up to 20, sha256('v5s1|sid|ctx|call') "
                       "order, distinct sessions first; branch order per state random (seed 504); replicate = first 15 by "
                       "sha256('v5s1rep|state_id')", "available": avail, "excluded": excl, "states": states},
              open(V5 / "configs/s1_states.json", "w"), indent=1)
    print(dict(avail), dict(excl), len(states))


# ----------------------------------------------------------------------------- history edits
def handle_defs(messages):
    """Programmatic handle definitions from the observed tool calls/results (no LLM)."""
    defs, pending = {}, []
    for m in messages:
        if m["role"] == "assistant":
            pending = G.parse_tool_calls(m["content"])[0][: G.MAX_TOOL_CALLS_PER_TURN]
        elif m["role"] == "tool" and pending:
            name, args = pending.pop(0)
            try:
                res = json.loads(m["content"].split("\nNote:")[0])
            except json.JSONDecodeError:
                continue
            if name == "filter" and isinstance(res, dict) and "handle" in res:
                defs[res["handle"]] = {"source": args.get("source"), "conditions": args.get("conditions"), "rows": res["rows"]}
    return defs


def current_request_start(messages):
    return max(i for i, m in enumerate(messages) if m["role"] == "user" and "[Request" in m["content"])


def edit_remove_repeats(messages):
    """Inside the current request, drop assistant turns whose text equals an earlier assistant turn in the same
    request, together with their following tool/environment messages. First occurrences are kept."""
    i0 = current_request_start(messages)
    head, cur = messages[: i0 + 1], messages[i0 + 1:]
    seen, out, skip = set(), [], False
    for m in cur:
        if m["role"] == "assistant":
            skip = m["content"] in seen
            seen.add(m["content"])
            if skip:
                continue
            out.append(m)
        elif skip and (m["role"] == "tool" or (m["role"] == "user" and m["content"].startswith("[Environment]"))):
            continue
        else:
            skip = False
            out.append(m)
    return head + out


def edit_state_summary(messages, st, sess):
    """System + one user message: previous requests' submitted answers, current handle definitions,
    current request text, last tool result. Built from observable state only."""
    i0 = current_request_start(messages)
    prev = [f"Request {r['request'] + 1}: {sess['requests'][r['request']]['text']} -> "
            + (f"submitted {json.dumps(r['answer'])}" if r["submitted"] else "no answer submitted (budget exhausted)")
            for r in st["req_results"]]
    defs = handle_defs(messages)
    hd = [f"{h}: from {d['source']} where {json.dumps(d['conditions'])} ({d['rows']} rows)" for h, d in defs.items()]
    last_tool = next((m["content"] for m in reversed(messages) if m["role"] == "tool"), "(none)")
    body = ("[State summary]\nPrevious requests:\n" + ("\n".join(prev) if prev else "(none)")
            + "\nExisting handles ('table' is always the original table):\n" + ("\n".join(hd) if hd else "(none)")
            + "\nLast tool result:\n" + last_tool + "\n\n" + messages[i0]["content"])
    return [messages[0], {"role": "user", "content": body}]


def edit_notice(messages):
    m = copy.deepcopy(messages)
    for x in reversed(m):
        if x["role"] == "tool":
            x["content"] = x["content"] + NOTICE
            break
    return m


# ----------------------------------------------------------------------------- run
def rebuild_state(split, s, atok, rtok):
    sess = DEV[s["session_id"]] if split == "dev" else TRAIN[s["session_id"]]
    cs = paths()[(split, s["session_id"], s["context"])]
    st, prompts, _ = SB.rebuild(atok, rtok, sess, s["context"], cs, s["state_call"])
    src = next(c for c in cs if c["call"] == s["state_call"])
    return sess, cs, st, prompts, src


def check():
    atok, rtok = SB.tokenizers()
    S = json.load(open(V5 / "configs/s1_states.json"))["states"]
    bad = 0
    for s in S:
        sess, cs, st, prompts, src = rebuild_state(s["split"], s, atok, rtok)
        bad += RV.sha(prompts[s["state_call"]]) != src["prompt_sha"]
    print("states", len(S), "prompt hash mismatches", bad)


def run(smoke=0):
    import httpx
    cfg = json.load(open(ROOT / "configs/models.json"))
    http = httpx.Client()
    servers = {m: Server(m, cfg["models"][m], http) for m in ("A", "B", "R")}
    for srv in servers.values():
        w = srv.render_ids([{"role": "user", "content": "Write a Python function that adds two numbers."}])
        for _ in range(3):
            srv.complete(w, max_tokens=32)
    plan = json.load(open(V5 / "configs/s1_states.json"))["states"]
    if smoke:
        plan = sorted(plan, key=lambda s: sha("v5s1smoke|" + s["state_id"]))[:smoke]
    rd = V5 / "results/stage1" / ("smoke" if smoke else "run")
    rd.mkdir(parents=True, exist_ok=True)
    out = {"run_id": rd.name, "fh": open(rd / "calls.jsonl", "a"), "sfh": open(rd / "sessions.jsonl", "a")}
    afh = open(rd / "restore_audit.jsonl", "a")
    done = set()
    for l in open(rd / "sessions.jsonl"):
        j = json.loads(l)
        done.add((j["state_id"], j["branch"], j["rep"]))
    units = [(s, b, 0) for s in plan for b in s["order"]]
    if not smoke:
        units += [(s, b, 1) for s in plan if s["replicate"] for b in s["order"]]
    nprep = 0
    for i, (s, br, rep) in enumerate(units):
        if (s["state_id"], br, rep) in done:
            continue
        sess, cs, st, prompts, src = rebuild_state(s["split"], s, servers["A"].tok, servers["R"].tok)
        ok_hash = RV.sha(prompts[s["state_call"]]) == src["prompt_sha"]
        tool_hash = SB.env_hash(st["env"])
        for srv in servers.values():
            srv.reset_cache()
        time.sleep(0.2)
        t0 = time.perf_counter()
        for c in cs:  # restore A cache with the logged prompts before the state (always_A path: B and R are empty)
            if c["call"] >= s["state_call"]:
                break
            servers["A"].complete(prompts[c["call"]], max_tokens=1)
            nprep += 1
        prep_s = time.perf_counter() - t0
        time.sleep(0.3)
        orig_ids = prompts[s["state_call"]]
        if br == "a2":
            st["messages"][:] = edit_remove_repeats(st["messages"])
        elif br == "a3":
            st["messages"][:] = edit_state_summary(st["messages"], st, sess)
        elif br == "a4":
            st["messages"][:] = edit_notice(st["messages"])
        m = BRANCH_MODEL[br]
        first_ids = RV.render(servers[m], st["messages"], "T1")
        expected = st["tracker"].predicted_hit(m, first_ids)
        tag = {"state_id": s["state_id"], "split": s["split"], "branch": br, "rep": rep, "state_request": s["request"],
               "prev_requests_correct": [r["correct"] for r in st["req_results"]]}
        summ, calls = run_branch(sess, s["context"], servers, out, st, m, tag)
        first = calls[0]
        afh.write(json.dumps({"state_id": s["state_id"], "branch": br, "rep": rep, "prompt_hash_ok": ok_hash,
                              "tool_state_hash": tool_hash, "history_edited": br in ("a2", "a3", "a4"),
                              "first_prompt_tokens": len(first_ids), "orig_prompt_tokens": len(orig_ids),
                              "expected_first_hit": expected, "actual_first_hit": first["cached_tokens"],
                              "first_hit_ok": expected == first["cached_tokens"], "prep_s": prep_s}) + "\n")
        afh.flush()
        print(f"[{i+1}/{len(units)}] {s['state_id']} {br} r{rep} {m} cur_req_ok={summ['requests'][s['request']]['correct']} "
              f"q={summ['q']:.2f} calls={summ['n_calls']} lat={summ['latency_s']:.1f}s hit={expected}/{first['cached_tokens']} "
              f"hash={ok_hash} prep_calls={nprep}", flush=True)


def run_branch(sess, ctx, servers, out, st, model, tag):
    """Force `model` for the rest of the current request (no router call), then pi0 = follow_R1 (T1)."""
    return _episode(sess, ctx, servers, out, st, model, tag, st["k"])


def _episode(sess, ctx, servers, out, st, model, tag, k0):
    """runner_v4.run_episode with: forced `model` while st['k'] == k0 (router not called), pi0 afterwards."""
    env, tracker, messages = st["env"], st["tracker"], st["messages"]
    calls = []
    t0 = time.perf_counter()
    while st["k"] < len(sess["requests"]):
        k = st["k"]
        if not st["request_open"]:
            messages.append({"role": "user", "content": st["notice"] + G.request_message(sess, k)})
            st["notice"], st["n_in_req"], st["request_open"] = "", 0, True
            st["submitted"], st["answer"] = False, None
        while not st["submitted"] and st["n_in_req"] < G.MAX_CALLS_PER_REQUEST:
            rec = {"run": out["run_id"], "session_id": sess["session_id"], "policy": "s1_branch", "context": ctx,
                   "schema": "T1", "call": st["call_idx"], "request": k, "call_in_request": st["n_in_req"],
                   "active_before": st["current"], **tag}
            rr = None
            if k == k0:
                chosen = model
            else:
                rr = iface.route("R1", servers["R"], messages, st["current"])
                chosen = rr["choice"]
            rec["router"] = rr and {k2: rr[k2] for k2 in ("choice", "parse_error", "raw", "input_tokens", "router_s")}
            rec["chosen"], rec["switch"] = chosen, st["current"] is not None and chosen != st["current"]
            tr0 = time.perf_counter()
            ids = RV.render(servers[chosen], messages, "T1")
            render_s = time.perf_counter() - tr0
            rec["pred_hit"] = {m: tracker.predicted_hit(m, ids) for m in ("A", "B")}
            c = servers[chosen].complete(ids, max_tokens=RV.MAX_TOKENS)
            tracker.add(chosen, ids)
            text = c["text"]
            messages.append({"role": "assistant", "content": text})
            tool_calls, perr = G.parse_tool_calls(text)
            results = []
            te = time.perf_counter()
            for name, args in tool_calls[: G.MAX_TOOL_CALLS_PER_TURN]:
                res = env.call(name, args)
                results.append({"name": name, "error": res.get("error"), "warning": res.get("warning"), "args": args})
                messages.append({"role": "tool", "content": json.dumps(res)})
                if name == "submit" and "error" not in res:
                    st["submitted"], st["answer"] = True, args.get("answer") if isinstance(args, dict) else None
                    break
            if not tool_calls:
                messages.append({"role": "user", "content": "[Environment] No valid tool call found"
                                 + (f" ({perr[0]})" if perr else "") + ". Respond with a tool call."})
            exec_s = time.perf_counter() - te
            rec.update({"prompt_tokens": len(ids), "prompt_sha": RV.sha(ids), "cached_tokens": c["cached_tokens"],
                        "completion_tokens": c["completion_tokens"], "ttft_s": c["ttft_s"], "total_s": c["total_s"],
                        "finish": c["finish_reason"], "error": c["error"], "render_s": render_s, "tool_exec_s": exec_s,
                        "n_tool_calls": len(tool_calls), "parse_errors": perr, "tools": results, "assistant_text": text,
                        "router_s": rr["router_s"] if rr else 0.0})
            rec["call_latency_s"] = rec["router_s"] + render_s + c["total_s"] + exec_s
            out["fh"].write(json.dumps(rec) + "\n"); out["fh"].flush()
            calls.append(rec)
            st["current"] = chosen
            st["call_idx"] += 1
            st["n_in_req"] += 1
        ok = st["submitted"] and G.correct(st["answer"], sess["expected"][k])
        st["req_results"].append({"request": k, "kind": sess["requests"][k]["kind"], "submitted": st["submitted"],
                                  "answer": st["answer"], "expected": sess["expected"][k], "correct": bool(ok),
                                  "calls": st["n_in_req"]})
        if not st["submitted"]:
            st["notice"] = "[Environment] The call budget for the previous request was exhausted.\n"
        st["request_open"] = False
        st["k"] += 1
    rr_ = st["req_results"]
    summ = {"run": out["run_id"], "session_id": sess["session_id"], "context": ctx, "policy": "s1_branch", "schema": "T1",
            "requests": rr_, "q": sum(r["correct"] for r in rr_) / len(rr_), "S": all(r["correct"] for r in rr_),
            "n_calls": len(calls), "latency_s": sum(c["call_latency_s"] for c in calls), "wall_s": time.perf_counter() - t0,
            "router_calls": sum(c["router"] is not None for c in calls), "switches": sum(c["switch"] for c in calls),
            "model_calls": {m: sum(c["chosen"] == m for c in calls) for m in "AB"}, **tag}
    out["sfh"].write(json.dumps(summ) + "\n"); out["sfh"].flush()
    return summ, calls


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    {"candidates": lambda: print(len(candidates()[0])), "select": select, "check": check,
     "run": lambda: run(a.smoke)}[a.cmd]()
