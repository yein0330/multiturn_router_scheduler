#!/bin/bash
# Waits for GPUs 1-3 to be free, then runs S1 full and S2 scoring under flock; records real exit codes.
cd ~/multiturn_routing_pilot
ST=followup_v5/results/WAIT_STATUS2
free() { local b=0; for g in 1 2 3; do b=$((b+$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l))); done; [ "$b" -eq 0 ]; }
step() {  # $1 name, $2 roles, rest command; retries while the lock runner aborts for busy GPUs
  local name=$1 roles=$2; shift 2
  while true; do
    until free; do sleep 120; done
    echo "$(date '+%F %T') start $name" >> $ST
    followup_v3/run_locked.sh "$roles" "$@"; rc=$?
    if tail -3 followup_v3/results/locked_runs.log | grep -q "ABORT GPU"; then
      echo "$(date '+%F %T') $name aborted (GPU busy), waiting" >> $ST; sleep 120; continue; fi
    echo "$(date '+%F %T') $name rc=$rc" >> $ST; break
  done
}
echo "$(date '+%F %T') waiter2 started" > $ST
step S1 "A B R" .venv/bin/python followup_v5/src/s1.py run
step S2score "R" .venv/bin/python followup_v5/src/s2.py score
echo "$(date '+%F %T') ALL_DONE" >> $ST
