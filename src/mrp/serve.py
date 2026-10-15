"""Start/stop the vLLM servers in configs/models.json (adapted from ~/switchcost; that repo is untouched).

    python -m mrp.serve start configs/models.json [A B R]
    python -m mrp.serve stop  configs/models.json
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]


def build_cmd(cfg, role):
    m = cfg["models"][role]
    if m.get("local_path"):
        target = [m["local_path"], "--served-model-name", m["served_model_name"]]
    else:
        target = [m["hf_id"], "--revision", m["revision"]]
    cmd = [str(ROOT / ".venv/bin/vllm"), "serve", *target,
           "--port", str(m["port"]), "--host", "127.0.0.1", "--max-model-len", str(m["max_model_len"])]
    for k, v in cfg["serve_flags"].items():
        if v is True:
            cmd.append(f"--{k}")
        elif v not in (False, None):
            cmd += [f"--{k}", str(v)]
    return cmd


def start(cfg, roles):
    (ROOT / "logs").mkdir(exist_ok=True)
    pid_file = ROOT / "logs" / "server_pids.json"
    pids = json.loads(pid_file.read_text()) if pid_file.exists() else {}
    for role in roles:
        log = open(ROOT / "logs" / f"vllm_{role}.log", "w")
        cmd = build_cmd(cfg, role)
        log.write("# CMD: " + " ".join(cmd) + "\n"); log.flush()
        env = dict(os.environ, **cfg["env"], CUDA_VISIBLE_DEVICES=str(cfg["models"][role]["gpu"]))
        pids[role] = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True).pid
    pid_file.write_text(json.dumps(pids))
    for role in roles:
        url = f"http://127.0.0.1:{cfg['models'][role]['port']}/health"
        t0 = time.time()
        while time.time() - t0 < 900:
            try:
                if httpx.get(url, timeout=2).status_code == 200:
                    print(f"{role} ready after {time.time()-t0:.0f}s"); break
            except httpx.HTTPError:
                pass
            time.sleep(3)
        else:
            sys.exit(f"{role} did not become healthy; see logs/vllm_{role}.log")


def stop(cfg):
    f = ROOT / "logs" / "server_pids.json"
    if f.exists():
        for role, pid in json.loads(f.read_text()).items():
            try:
                os.killpg(pid, signal.SIGINT); print(f"SIGINT {role} ({pid})")
            except ProcessLookupError:
                pass
        f.unlink()


if __name__ == "__main__":
    action, path = sys.argv[1], sys.argv[2]
    cfg = json.loads(Path(path).read_text())
    if action == "start":
        start(cfg, tuple(sys.argv[3:]) or ("A", "B", "R"))
    else:
        stop(cfg)
