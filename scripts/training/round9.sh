#!/usr/bin/env bash
# Round 9: deeper voice model (acoustic stack 5 -> 8 blocks, 4.54M -> 5.73M parameters), branched from the
# release (voice-four 135k, EMA weights) with the new blocks starting as identity. Same data as round 8.
cd "$(dirname "$0")/.."
LIB=${EXTRA_LIBS_DIR:-$HOME/.local/share/tiny-voca-libs}
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/round9.log; }
governor(){ .venv/bin/python scripts/gpu_governor.py --file $1 --max-util 85 --target-util 80 --max-power 250 --start 0 >> $1.log 2>&1 & echo $!; }
if ! grep -q 'voice-five done' logs/round9.log 2>/dev/null; then
  START=(--init-from runs/voice-four/step-00135000-ema.pt --new-data --identity-new-blocks); [ -f runs/voice-five/last.pt ] && START=(--resume)
  log "voice-five: start (${START[*]})"; G=$(governor logs/voice-five.throttle.json)
  PYTHONPATH=$LIB/pylib .venv/bin/python -m tiny_voca.train "${START[@]}" \
    --train data/em4-annotations/train.jsonl data/ef4-annotations/train.jsonl data/emotion4-annotations/train-x3.jsonl \
            data/it-multi6-annotations/train-it1.jsonl data/emotion-annotations/train-no-ef3-neutral.jsonl \
    --validation data/em4-annotations/validation.jsonl data/ef4-annotations/validation.jsonl data/emotion4-annotations/validation.jsonl \
                 data/it-multi6-annotations/validation-it1.jsonl data/emotion-annotations/validation.jsonl \
    --validation-size 96 --output runs/voice-five --config configs/italian-voice-round-9.json --steps 155000 --batch-frames 9000 --max-batch 32 \
    --learning-rate 1e-4 --warmup 300 --gan-start 0 --gan-ramp 1000 --gan-ratio 0.7 --teacher-hold 0 --teacher-ramp 1 --teacher-final 0 \
    --weights '{"pitch_spread": 4.0}' --pitch-delta 10 --lr-schedule cosine --ema-decay 0.999 --stream \
    --throttle-file logs/voice-five.throttle.json --save-every 1000 --keep-every 1000 --max-seconds 172800 >> logs/voice-five.log 2>&1
  kill $G 2>/dev/null
  [ -f runs/voice-five/inference.pt ] && grep -q '"step": 155000' runs/voice-five/metrics.jsonl 2>/dev/null || { log "voice-five failed"; exit 1; }
  log "voice-five done"
fi
log "round 9 done"
