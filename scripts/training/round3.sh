#!/bin/bash
# Round 3 (one GPU job at a time; each stage resumable and skipped when done):
#   1. wait for the adversarial pilot (runs/voice-adv-pilot, 103k -> 113k)
#   2. soft-gated fine-tune (unvoiced_pulse_gain 0.5) with light adversarial training: runs/voice-gate, 113k -> 128k
#   3. vocoder pairs from the gated voice model, on the GPU: data/vocoder-pairs-gate
#   4. prior vocoder from 150k on the new pairs: runs/vocoder-prior2, 150k -> 180k
cd "$(dirname "$0")/../.."
export PYTHONPATH=$EXTRA_LIBS_DIR/pylib
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/round3.log; }
step_of(){ .venv/bin/python -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=True)['step'])" "$1" 2>/dev/null || echo 0; }
running(){ pgrep -f "^[^ ]*python[^ ]* $1" > /dev/null; }
governor(){ .venv/bin/python scripts/gpu_governor.py --file $1 --max-util 85 --target-util 80 --max-power 250 --start 0 >> $1.log 2>&1 & echo $!; }

until grep -q "PILOT-EXIT" logs/voice-adv-pilot.log; do sleep 60; done
log "adversarial pilot done at $(step_of runs/voice-adv-pilot/last.pt)"

if [ "$(step_of runs/voice-gate/last.pt)" -lt 128292 ]; then
  START=(--init-from runs/voice-adv-pilot/last.pt); [ -f runs/voice-gate/last.pt ] && START=(--resume runs/voice-gate/last.pt)
  log "voice-gate: start (${START[*]})"; G=$(governor logs/voice-gate.throttle.json)
  .venv/bin/python -m tiny_voca.train "${START[@]}" \
    --train data/it-multi6-annotations/train.jsonl data/emotion-annotations/train.jsonl \
    --validation data/it-multi6-annotations/validation.jsonl data/emotion-annotations/validation.jsonl --validation-size 84 \
    --output runs/voice-gate --config configs/italian-voice-emotion-gate.json --steps 128292 --batch-frames 9000 --max-batch 32 \
    --learning-rate 1e-4 --warmup 200 --gan-start 0 --gan-ramp 500 --gan-ratio 0.5 --teacher-hold 0 --teacher-ramp 1 --teacher-final 0 \
    --weights '{"pitch_spread": 4.0}' --lr-schedule cosine --ema-decay 0.999 --stream \
    --throttle-file logs/voice-gate.throttle.json --save-every 1000 --keep-every 5000 --max-seconds 172800 >> logs/voice-gate.log 2>&1
  kill $G 2>/dev/null
  [ "$(step_of runs/voice-gate/last.pt)" -ge 128292 ] || { log "voice-gate stopped early at $(step_of runs/voice-gate/last.pt)"; exit 1; }
fi
log "voice-gate done"

if ! grep -q '"complete": true' logs/pairs-gate.log 2>/dev/null; then
  log "pairs: start (GPU)"
  .venv/bin/python -m tiny_voca.vocoder_pairs --checkpoint runs/voice-gate/inference.pt --device cuda \
    --annotations data/it-multi6-annotations/train.jsonl data/emotion-annotations/train.jsonl --output data/vocoder-pairs-gate/train >> logs/pairs-gate.log 2>&1
  grep -q '"complete": true' logs/pairs-gate.log || { log "pairs failed"; exit 1; }
fi
log "pairs done"

if ! grep -q '"done"' logs/vocoder-prior2.log 2>/dev/null; then
  START=(--init-from runs/vocoder-prior/last.pt --fresh-schedule); [ -f runs/vocoder-prior2/last.pt ] && START=(--resume)
  log "vocoder-prior2: start"; G=$(governor logs/vocoder-prior2.throttle.json)
  .venv/bin/python -m tiny_voca.train_vocoder "${START[@]}" --prior --manifests data/it-multi-features/train.jsonl \
    --pairs data/vocoder-pairs-gate/train/pairs.jsonl --pair-fraction 0.5 --validation data/it1-full-features/test.jsonl \
    --output runs/vocoder-prior2 --steps 180000 --batch 64 --learning-rate 1.5e-4 --warmup 500 \
    --teacher vocos --distill-mode spectrum --distill-weight 4 --keep-every 10000 --throttle-file logs/vocoder-prior2.throttle.json >> logs/vocoder-prior2.log 2>&1
  kill $G 2>/dev/null
  grep -q '"done"' logs/vocoder-prior2.log || { log "vocoder-prior2 stopped early"; exit 1; }
fi
log "round 3 done"
