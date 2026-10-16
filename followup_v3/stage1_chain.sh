#!/bin/bash
cd ~/multiturn_routing_pilot
P=.venv/bin/python
$P followup_v3/src/profile_v3.py measure > followup_v3/results/stage01/profile_measure.log 2>&1 || exit 11
$P followup_v3/src/agent_runner.py --sessions followup_v3/data/dev.jsonl --run-id dev50 > followup_v3/results/dev50_run.log 2>&1 || exit 12
