#!/bin/bash
# Watchdog for the alternating GPU schedule. Fixes what is safe to fix, exits (= alert) on anything else.
cd "$(dirname "$0")/../.."
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/watchdog.log; }
alert(){ log "ALERT: $*"; echo "ALERT: $*"; exit 1; }
progress(){  # a number that grows while any block works
  local a=$(cat runs/voice/metrics.jsonl runs/voice-emotion/metrics.jsonl 2>/dev/null | wc -l)
  local b=$(cat runs/vocoder/metrics.jsonl runs/vocoder-distill/metrics.jsonl runs/vocoder-ft/metrics.jsonl runs/vocoder-prior/metrics.jsonl data/vocoder-pairs/train/pairs.jsonl 2>/dev/null | wc -l)
  local m=$(cat data/emotion-synth/manifest.jsonl data/emotion-synth/rejected.jsonl 2>/dev/null | wc -l)
  echo $(( ${a:-0} + ${b:-0} + ${m:-0} ))
}
declare -A seen; restarts=0; full=0; last=$(progress); changed=$(date +%s)
log "watchdog started"
while true; do
  sleep 120
  if grep -q "all blocks done" logs/alternate.log; then log "schedule finished"; echo "DONE: all blocks finished"; exit 0; fi
  if ! pgrep -f "bin/bash logs/alternate.sh" > /dev/null; then
    restarts=$((restarts + 1)); [ $restarts -gt 3 ] && alert "alternation keeps dying ($(tail -2 logs/alternate.log | tr '\n' ' '))"
    log "alternation not running, restart $restarts: $(tail -1 logs/alternate.log)"
    ( setsid nohup logs/alternate.sh >> logs/alternate.log 2>&1 < /dev/null & )
    sleep 120
  fi
  now=$(progress)
  if [ "$now" != "$last" ]; then last=$now; changed=$(date +%s)
  elif [ $(( $(date +%s) - changed )) -gt 1200 ]; then alert "no progress for 20 minutes (last: $(tail -1 logs/alternate.log))"; fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -gt 15500 ]; then full=$((full + 1)); [ $full -ge 3 ] && alert "GPU memory at ${used} MiB for 6 minutes"; else full=0; fi
  for f in logs/voice.log logs/vocoder.log logs/emotion-gen.log logs/emotion-prep.log logs/vocoder-distill.log logs/vocoder-ft.log logs/pairs.log logs/vocoder-prior.log; do
    n=$(grep -c Traceback "$f" 2>/dev/null); n=${n:-0}
    [ -z "${seen[$f]}" ] && seen[$f]=$n
    [ "$n" -gt "${seen[$f]}" ] && alert "new crash in $f: $(grep -A12 Traceback "$f" | grep -i error | tail -1)"
  done
  if ! curl -s -m 10 localhost:8790/info > /dev/null; then log "playground down, restarting"; ( setsid nohup logs/playground.sh > logs/playground.log 2>&1 < /dev/null & ); fi
  freec=$(df --output=avail -BG /mnt/c | tail -1 | tr -dc 0-9)
  [ "$freec" -lt 20 ] && alert "Windows C: has only ${freec} GB free"
done
