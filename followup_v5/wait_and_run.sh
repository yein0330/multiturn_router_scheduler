#!/bin/bash
# Waits until GPUs 1-3 have no compute processes (polls every 120 s), then runs S1 smoke and S2 router scoring
# under the flock runner. Writes followup_v5/results/WAIT_STATUS.
cd ~/multiturn_routing_pilot
ST=followup_v5/results/WAIT_STATUS
echo "$(date '+%F %T') waiting for GPUs 1-3" > $ST
while true; do
  busy=0
  for g in 1 2 3; do n=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l); busy=$((busy+n)); done
  [ "$busy" -eq 0 ] && break
  sleep 120
done
echo "$(date '+%F %T') GPUs free; S1 smoke" >> $ST
followup_v3/run_locked.sh "A B R" .venv/bin/python followup_v5/src/s1.py run --smoke 3 > followup_v5/results/stage1/smoke.log 2>&1
echo "$(date '+%F %T') S1 smoke rc=$?" >> $ST
followup_v3/run_locked.sh "R" .venv/bin/python followup_v5/src/s2.py score > followup_v5/results/stage2/score.log 2>&1
echo "$(date '+%F %T') S2 score rc=$? DONE" >> $ST
