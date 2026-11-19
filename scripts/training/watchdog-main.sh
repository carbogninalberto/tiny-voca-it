#!/usr/bin/env bash
# Permanent supervisor (never exits): every 60 s
#  - writes an EVENT line to logs/events.log for every new line in the round logs (stage start/done/failed),
#  - ALERT on new Tracebacks in any training/generation log, GPU memory > 15.5 GB for 3 minutes, RAM < 1.5 GB,
#  - restarts a round script that died before writing its "done" line (rounds listed in logs/active-rounds),
#  - keeps the playground alive.
cd "$(dirname "$0")/.."
ev(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/events.log; }
declare -A seen; high=0
tb(){ ls logs/*.log | grep -v -E "events.log|watchdog.log|playground.log" | xargs cat 2>/dev/null | grep -c Traceback; }
tbs=$(tb)
while true; do
  for f in logs/round*.log; do
    n=$(wc -l < "$f"); old=${seen[$f]:-$n}
    [ "$n" -gt "$old" ] && tail -n $((n - old)) "$f" | while read -r line; do ev "EVENT $(basename $f .log): $line"; done
    seen[$f]=$n
  done
  now=$(tb); [ "$now" -gt "$tbs" ] && ev "ALERT new Traceback in logs"; tbs=$now
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$mem" -gt 15500 ]; then high=$((high + 1)); [ $high -eq 3 ] && ev "ALERT GPU memory $mem MiB for 3 minutes"; else high=0; fi
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); [ "$avail" -lt 1500 ] && ev "ALERT RAM available $avail MiB"
  for r in $(cat logs/active-rounds 2>/dev/null); do
    if ! ps -eo args | awk -v s="logs/$r.sh" '$2==s {f=1} END {exit !f}' && ! grep -q -E "$r done|round [0-9]+ done" logs/$r.log 2>/dev/null; then
      tail -1 logs/$r.log | grep -q failed && { ev "ALERT $r stopped: $(tail -1 logs/$r.log)"; continue; }
      ev "EVENT restarting $r"; ( setsid nohup logs/$r.sh > /dev/null 2>&1 < /dev/null & )
    fi
  done
  curl -s -m 10 localhost:8790/info > /dev/null || ( setsid nohup logs/playground.sh > logs/playground.log 2>&1 < /dev/null & )
  sleep 60
done
