#!/bin/bash
# followup_v2 GPU chain. Single owner: whole chain runs under an exclusive flock.
cd ~/multiturn_routing_pilot
LOG=followup_v2/results/chain_status.log
mkdir -p followup_v2/results
exec 9>.gpu_runner.lock
if ! flock -n 9; then echo "$(date '+%F %T') ABORT: lock held by another runner" >> $LOG; exit 1; fi
st() { echo "$(date '+%F %T') $*" >> $LOG; }
for g in 1 2; do
  n=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$n" -gt 0 ]; then st "ABORT: GPU $g busy (foreign process)"; exit 1; fi
done
export PYTHONPATH=src
.venv/bin/python -m mrp.serve start configs/models.json A B >> $LOG 2>&1 || { st "ABORT: server start failed"; exit 1; }
st "servers A,B up (started by this chain)"
P="../../.venv/bin/python"
cd followup_v2/src
$P v2_main.py baseline >> ../results/baseline_check.log 2>&1; st "baseline exit=$?"
$P v2_main.py decide   >> ../results/baseline_check.log 2>&1; st "decide exit=$?"
$P v2_main.py main     >> ../results/main.log 2>&1;           st "main exit=$?"
$P v2_main.py branch   >> ../results/branch.log 2>&1;         st "branch exit=$?"
cd ../..
.venv/bin/python -m mrp.serve stop configs/models.json >> $LOG 2>&1
st "servers stopped; CHAIN_DONE"
