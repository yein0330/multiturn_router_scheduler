#!/bin/bash
# Runs stage B2 after stage A finishes (scope frozen in design.md §12.1).
cd ~/multiturn_routing_pilot
export PYTHONPATH=src
while pgrep -f "run-id eval[A]" >/dev/null; do sleep 20; done
R=".venv/bin/python -m mrp.runner --sessions data/eval/eval.jsonl --cost-fit profiles/fit.json"
$R --policies P1,P2,P4,P6,P7,P8 --context long --cache retain --run-id evalB_long_retain
$R --policies P4,P6,P7,P8 --context short --cache evict --run-id evalB_short_evict
$R --policies P4,P6,P7,P8 --context long --cache evict --run-id evalB_long_evict
echo STAGE_B_DONE
