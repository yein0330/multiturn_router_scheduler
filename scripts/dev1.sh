#!/bin/bash
set -e
cd ~/multiturn_routing_pilot
export PYTHONPATH=src
R=".venv/bin/python -m mrp.runner"
$R --sessions data/dev/calib_single.jsonl --policies P1,P2,P4 --run-id dev_calib_single
$R --sessions data/dev/smoke.jsonl --policies P1,P2,P3,P4,P5,P6 --run-id dev_smoke
$R --sessions data/dev/calib_sessions.jsonl --policies P1,P2,P4 --run-id dev_calib_sessions
