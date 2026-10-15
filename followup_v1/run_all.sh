#!/bin/bash
# Overnight follow-up chain. Single GPU owner: waits for the B2 rerun (scripts/evalB.sh) to exit first.
cd ~/multiturn_routing_pilot
export PYTHONPATH=src
LOG=followup_v1/results/chain_status.log
B2PID=${1:-1191556}
echo "$(date '+%F %T') waiting for B2 pid $B2PID" >> $LOG
while kill -0 $B2PID 2>/dev/null; do sleep 30; done
if pgrep -f "mrp\.runne[r]|fu_runne[r]|branch\.p[y]" >/dev/null; then echo "$(date '+%F %T') ABORT: another runner is active" >> $LOG; exit 1; fi
echo "$(date '+%F %T') B2 exited; counts: $(wc -l results/evalB_*/sessions.jsonl | tr '\n' ' ')" >> $LOG
# 2) new policies + drift reruns
cd followup_v1/src
PYTHONPATH=../../src ../../.venv/bin/python fu_runner.py --run-id newpol >> ../results/newpol.log 2>&1
echo "$(date '+%F %T') newpol exit=$? rows=$(wc -l < ../results/newpol/sessions.jsonl)" >> ../../$LOG
# 3A) initial-choice branch diagnostic
PYTHONPATH=../../src ../../.venv/bin/python branch.py run ../configs/points_init.json ../results/branch_init >> ../results/branch_init.log 2>&1
echo "$(date '+%F %T') branch_init exit=$? rows=$(wc -l < ../results/branch_init/measurements.jsonl)" >> ../../$LOG
# 3B) switch diagnostic only if B2 complete
cd ../..
n1=$(wc -l < results/evalB_long_retain/sessions.jsonl 2>/dev/null || echo 0)
n2=$(wc -l < results/evalB_short_evict/sessions.jsonl 2>/dev/null || echo 0)
n3=$(wc -l < results/evalB_long_evict/sessions.jsonl 2>/dev/null || echo 0)
if [ "$n1" -eq 300 ] && [ "$n2" -eq 200 ] && [ "$n3" -eq 200 ]; then
  cd followup_v1/src
  PYTHONPATH=../../src ../../.venv/bin/python branch.py select-switch evalB_long_retain,evalB_short_evict,evalB_long_evict ../configs/points_switch.json >> ../results/branch_switch.log 2>&1
  PYTHONPATH=../../src ../../.venv/bin/python branch.py run ../configs/points_switch.json ../results/branch_switch >> ../results/branch_switch.log 2>&1
  echo "$(date '+%F %T') branch_switch exit=$? rows=$(wc -l < ../results/branch_switch/measurements.jsonl)" >> ../../$LOG
else
  echo "$(date '+%F %T') SKIP branch_switch: B2 incomplete ($n1/300, $n2/200, $n3/200)" >> $LOG
fi
echo "$(date '+%F %T') CHAIN_DONE" >> ~/multiturn_routing_pilot/$LOG
