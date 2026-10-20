#!/bin/bash
cd ~/multiturn_routing_pilot
P=.venv/bin/python
$P followup_v5/src/stage0.py 0a > followup_v5/results/stage0/0a.log 2>&1 || exit 31
$P followup_v5/src/stage0.py 0b 6000 > followup_v5/results/stage0/0b.log 2>&1 || exit 32
