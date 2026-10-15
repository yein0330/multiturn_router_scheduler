#!/bin/bash
set -e
cd ~/multiturn_routing_pilot
export PYTHONPATH=src
R=".venv/bin/python -m mrp.runner --sessions data/eval/eval.jsonl --cost-fit profiles/fit.json"
$R --policies P1,P2,P4,P6,P7,P8 --context long --cache retain --run-id evalB_long_retain
$R --policies P4,P6,P7,P8 --context short --cache evict --run-id evalB_short_evict
$R --policies P4,P6,P7,P8 --context long --cache evict --run-id evalB_long_evict
