"""Write v3 session files + manifest (seed, sha256). Split by domain + phrasing variant.
    python make_v3_data.py smoke|dev|train|eval
"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import agent_tasks as G
V3 = Path(__file__).resolve().parents[1]
SPEC = {"smoke": ("dev", 2, 61000), "dev": ("dev", 10, 62000), "train": ("train", 10, 63000), "eval": ("eval", 10, 65000)}

def main(which):
    split, per, base = SPEC[which]
    ss = [G.build_session(split, sc, i, base + 100 * j + i) for j, sc in enumerate(G.PATTERNS) for i in range(per)]
    for s in ss:
        s["session_id"] = f"{which}-{s['scenario']}-{s['index']:02d}"
    p = V3 / "data" / f"{which}.jsonl"
    p.parent.mkdir(exist_ok=True)
    with open(p, "w") as f:
        for s in ss: f.write(json.dumps(s) + "\n")
    man = {"file": str(p.relative_to(V3)), "split": split, "domains": G.SPLIT_DOMAINS[split], "variant": G.VARIANT[split],
           "n": len(ss), "seeds": [s["seed"] for s in ss], "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
           "knobs": {"patterns": G.PATTERNS, "rows_per_table": 80, "context": G.CONTEXT, "exposed_scenario": "expose",
                     "max_calls_per_request": G.MAX_CALLS_PER_REQUEST, "max_tool_calls_per_turn": G.MAX_TOOL_CALLS_PER_TURN}}
    json.dump(man, open(V3 / "data" / f"manifest_{which}.json", "w"), indent=1)
    print(which, len(ss), man["sha256"][:12])

if __name__ == "__main__":
    main(sys.argv[1])
