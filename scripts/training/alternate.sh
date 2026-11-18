#!/bin/bash
# Alternate the GPU, one job at a time. Done so far: vocoder 0->50k, voice 14k->64k (runs/voice), emotion data.
# Now: vocoder -> 100k, voice block 2 with emotion (runs/voice-emotion, 64k -> 114k), then the vocoder distilled from
# Vocos (runs/vocoder-distill, warm start from runs/vocoder at 100k; the plain run plateaued on a metallic residue).
# Voice blocks stop early when validation has not improved for 8 checks. Progress: logs/alternate.log
cd "$(dirname "$0")/../.."
log(){ echo "$(date '+%m-%d %H:%M') $*"; }
vocoder_pid(){ pgrep -f "^[^ ]*python[^ ]* -m tiny_voca[.]train_vocoder" | head -1; }
step_of(){ .venv/bin/python -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=True)['step'])" "$1" 2>/dev/null || echo 0; }

vocoder_block(){  # $1 = step to reach (a kept milestone)
  local name=$(printf "%07d" $1)
  [ -f runs/vocoder/keep-$name/last.pt ] && { log "vocoder: $1 already reached"; return; }
  if [ -z "$(vocoder_pid)" ]; then log "vocoder: resume"; RESUME=1 setsid nohup logs/vocoder.sh >> logs/vocoder.log 2>&1 < /dev/null & sleep 60; fi
  until [ -f runs/vocoder/keep-$name/last.pt ]; do
    [ -z "$(vocoder_pid)" ] && { log "vocoder stopped unexpectedly before $1"; tail -5 logs/vocoder.log; exit 1; }
    sleep 30
  done
  if [ "$1" -lt 300000 ]; then local p=$(vocoder_pid); kill "$p"; while kill -0 "$p" 2>/dev/null; do sleep 5; done; log "vocoder paused at $1"; fi
}

voice_block(){  # $1 = run directory, $2 = launch script, $3 = step to reach
  local run=$1 script=$2 target=$3
  local pattern="^[^ ]*python[^ ]* -m tiny_voca[.]train .*--output $run "
  rm -f logs/voice.stop
  local now=$(step_of $run/last.pt)
  [ "$now" -ge "$target" ] && { log "voice $run: $target already reached"; return; }
  if pgrep -f "$pattern" > /dev/null; then
    log "voice $run: already training, monitoring until step $target"
  else
    if [ -f $run/last.pt ]; then MODE=resume; else MODE=init; fi
    log "voice $run: $MODE, until step $target"
    MODE=$MODE setsid nohup $script >> logs/voice.log 2>&1 < /dev/null &
    sleep 180
  fi
  local verdict=run
  while pgrep -f "$pattern" > /dev/null; do
    verdict=$(.venv/bin/python - "$run" "$target" <<'PY'
import json, sys, os
run, target = sys.argv[1], int(sys.argv[2])
path = f"{run}/metrics.jsonl"
rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
step = rows[-1]["step"] if rows else 0
values = [r["validation_predicted_vocoder_input"] for r in rows if "validation_predicted_vocoder_input" in r]
if step >= target: print("target")
elif len(values) >= 12 and min(values[-8:]) > min(values): print("plateau")
else: print("run")
PY
)
    if [ "$verdict" != run ]; then log "voice $run: stop ($verdict)"; touch logs/voice.stop; break; fi
    sleep 60
  done
  while pgrep -f "$pattern" > /dev/null; do sleep 10; done
  local reached=$(step_of $run/last.pt)
  if [ "$verdict" = run ] && [ "$reached" -lt "$target" ]; then log "voice $run exited early at $reached (no stop requested)"; tail -3 logs/voice.log; exit 1; fi
  log "voice $run paused at $reached"
}

distill_block(){  # train the distilled vocoder to its configured end (logs/vocoder-distill.sh), resumable
  local pattern="^[^ ]*python[^ ]* -m tiny_voca[.]train_vocoder .*--output runs/vocoder-distill "
  if ! pgrep -f "$pattern" > /dev/null; then
    log "vocoder-distill: start/resume"
    ( setsid nohup logs/vocoder-distill.sh >> logs/vocoder-distill.log 2>&1 < /dev/null & )
    sleep 120
  fi
  while pgrep -f "$pattern" > /dev/null; do sleep 30; done
  grep -q '"done"' logs/vocoder-distill.log || { log "vocoder-distill stopped without finishing"; tail -5 logs/vocoder-distill.log; exit 1; }
  log "vocoder-distill: finished"
}

# Done before this point: vocoder 0->100k (runs/vocoder), voice blocks 1-2 (block 2 early-stopped at 103292), pairs.
prior_block(){  # harmonic-prior vocoder with the voice model's own spectrograms mixed in
  local pattern="^[^ ]*python[^ ]* -m tiny_voca[.]train_vocoder .*--output runs/vocoder-prior "
  if ! pgrep -f "$pattern" > /dev/null; then
    log "vocoder-prior: start/resume"
    ( setsid nohup logs/vocoder-prior.sh >> logs/vocoder-prior.log 2>&1 < /dev/null & )
    sleep 120
  fi
  while pgrep -f "$pattern" > /dev/null; do sleep 30; done
  grep -q '"done"' logs/vocoder-prior.log || { log "vocoder-prior stopped without finishing"; tail -5 logs/vocoder-prior.log; exit 1; }
  log "vocoder-prior: finished"
}

# Done: vocoder 0->100k, voice blocks 1-2, pairs, distillation (stopped at its last save: phase not learnable that way).
log "schedule: harmonic-prior vocoder (pairs mixed in)"
prior_block
log "all blocks done"
