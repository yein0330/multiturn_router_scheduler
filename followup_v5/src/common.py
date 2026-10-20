"""followup_v5 shared helpers: T1-min schema, data pools, paths. Reuses v3/v4 modules (imported, not modified)."""
import copy, hashlib, json, sys
from pathlib import Path

V5 = Path(__file__).resolve().parents[1]
ROOT = V5.parent
for p in (ROOT / "src", ROOT / "followup_v3/src", ROOT / "followup_v4/src", V5 / "src"):
    sys.path.insert(0, str(p))
import agent_tasks as G  # noqa: E402
import iface  # noqa: E402

# T1-min: T0 text descriptions unchanged; only structural constraints added (no new description text).
T1MIN = copy.deepcopy(G.TOOLS)
for t in T1MIN:
    f = t["function"]
    if f["name"] == "filter":
        f["parameters"]["properties"]["conditions"]["items"] = {
            "type": "object", "properties": {"column": {"type": "string"}, "op": {"type": "string", "enum": iface.FILTER_OPS},
                                             "value": {}}, "required": ["column", "op", "value"]}
    if f["name"] == "aggregate":
        f["parameters"]["properties"]["op"] = {"type": "string", "enum": iface.AGG_OPS}
iface.TOOLSETS["T1min"] = T1MIN


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def train_pool(n_per_scenario=20):
    """Extended train pool: same generator and seed rule as followup_v3 make_v3_data (63000 + 100*j + i);
    i < 10 reproduces the existing v3 train sessions exactly."""
    out = []
    for j, sc in enumerate(G.PATTERNS):
        for i in range(n_per_scenario):
            s = G.build_session("train", sc, i, 63000 + 100 * j + i)
            s["session_id"] = f"train-{sc}-{i:02d}"
            out.append(s)
    return out
