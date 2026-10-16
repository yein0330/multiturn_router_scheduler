#!/bin/bash
# usage: run_locked.sh "<roles to serve, e.g. R or A B>" <command...>
# Runs a GPU job under an exclusive flock; starts only the listed servers and stops only those.
cd ~/multiturn_routing_pilot
LOG=$PWD/followup_v3/results/locked_runs.log
exec 9>$PWD/.gpu_runner.lock
if ! flock -n 9; then echo "$(date '+%F %T') ABORT lock held" >> $LOG; exit 1; fi
ROLES="$1"; shift
declare -A GPU=([A]=1 [B]=2 [R]=3)
for r in $ROLES; do
  n=$(nvidia-smi -i ${GPU[$r]} --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$n" -gt 0 ]; then echo "$(date '+%F %T') ABORT GPU ${GPU[$r]} busy" >> $LOG; exit 1; fi
done
export PYTHONPATH=$PWD/src
.venv/bin/python -m mrp.serve start ${MODELS_CFG:-configs/models.json} $ROLES >> $LOG 2>&1 || { echo "$(date '+%F %T') ABORT serve" >> $LOG; exit 1; }
echo "$(date '+%F %T') START [$ROLES] $*" >> $LOG
"$@"; rc=$?
echo "$(date '+%F %T') END rc=$rc $*" >> $LOG
.venv/bin/python -m mrp.serve stop ${MODELS_CFG:-configs/models.json} >> $LOG 2>&1
echo "$(date '+%F %T') servers stopped" >> $LOG
exit $rc
