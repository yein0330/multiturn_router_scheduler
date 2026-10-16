#!/bin/bash
cd ~/multiturn_routing_pilot
P=.venv/bin/python
$P followup_v3/src/gate1.py measure-paired > followup_v3/results/stage01/paired_run.log 2>&1 || exit 21
$P followup_v3/src/stage1_rho.py > followup_v3/results/stage01/rho_run.log 2>&1 || exit 22
