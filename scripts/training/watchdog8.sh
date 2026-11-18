#!/usr/bin/env bash
# Watchdog for round 8: restart the (resumable) pipeline if it died, alert on tracebacks, GPU-memory
# spill, low RAM or disk; keep the playground alive; log usage every 10 minutes.
cd "$(dirname "$0")/.."
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/watchdog.log; }
alert(){ log "ALERT $*"; echo "ALERT $(date '+%H:%M') $*"; }
seen=$(cat logs/*-gen.log logs/round8-prep.log logs/voice-four.log logs/vocoder-four.log logs/emotion4-gen.log 2>/dev/null | grep -c Traceback)
high=0; tick=0
while ! grep -q 'round 8 done' logs/round8.log 2>/dev/null; do
  if ! pgrep -f "logs/round8.sh" > /dev/null; then
    if grep -q -E 'failed' <(tail -1 logs/round8.log); then alert "round 8 stopped: $(tail -1 logs/round8.log)"; sleep 600; continue; fi
    log "round 8 not running: restarting"; ( setsid nohup logs/round8.sh > /dev/null 2>&1 < /dev/null & )
  fi
  now=$(cat logs/*-gen.log logs/round8-prep.log logs/voice-four.log logs/vocoder-four.log logs/emotion4-gen.log 2>/dev/null | grep -c Traceback)
  [ "$now" -gt "$seen" ] && alert "new Traceback in round 8 logs"; seen=$now
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$mem" -gt 15500 ]; then high=$((high + 1)); [ $high -eq 3 ] && alert "GPU memory at $mem MiB for 6 minutes"; else high=0; fi
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); [ "$avail" -lt 1500 ] && alert "RAM available $avail MiB"
  free_c=$(df -BG --output=avail /mnt/c 2>/dev/null | tail -1 | tr -dc 0-9); [ -n "$free_c" ] && [ "$free_c" -lt 20 ] && alert "C: free ${free_c} GB"
  curl -s -m 10 localhost:8790/info > /dev/null || { log "playground down, restarting"; ( setsid nohup logs/playground.sh > logs/playground.log 2>&1 < /dev/null & ); }
  tick=$((tick + 1)); [ $((tick % 5)) -eq 0 ] && log "usage: RAM available $avail MiB, GPU MiB $mem, load $(cut -d' ' -f1 /proc/loadavg)"
  sleep 120
done
log "round 8 finished"
