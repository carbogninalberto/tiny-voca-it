#!/usr/bin/env bash
# Round 7: short voice fine-tune from voice-two 115k with the frame-to-frame pitch-change loss
# (against tremble, so the pitch spread comes from intonation). 6000 steps, checkpoints every 1000;
# the best checkpoint is chosen per voice on jitter, spread, mel error and locking. Resumable.
cd "$(dirname "$0")/.."
LIB=${EXTRA_LIBS_DIR:-$HOME/.local/share/tiny-voca-libs}
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/round7.log; }
governor(){ .venv/bin/python scripts/gpu_governor.py --file $1 --max-util 85 --target-util 80 --max-power 250 --start 0 >> $1.log 2>&1 & echo $!; }
if ! grep -q 'voice-three done' logs/round7.log 2>/dev/null; then
  START=(--init-from runs/voice-two/step-00115000.pt --new-data); [ -f runs/voice-three/last.pt ] && START=(--resume)
  log "voice-three: start (${START[*]})"; G=$(governor logs/voice-three.throttle.json)
  PYTHONPATH=$LIB/pylib .venv/bin/python -m tiny_voca.train "${START[@]}" \
    --train data/it-multi6-annotations/train-it1.jsonl data/ef3-annotations/train.jsonl data/emotion-annotations/train-no-ef3-neutral.jsonl \
    --validation data/it-multi6-annotations/validation-it1.jsonl data/ef3-annotations/validation.jsonl data/emotion-annotations/validation.jsonl \
    --validation-size 96 --output runs/voice-three --config configs/italian-voice-emotion.json --steps 121000 --batch-frames 9000 --max-batch 32 \
    --learning-rate 5e-5 --warmup 300 --gan-start 0 --gan-ramp 1000 --gan-ratio 0.7 --teacher-hold 0 --teacher-ramp 1 --teacher-final 0 \
    --weights '{"pitch_spread": 4.0}' --pitch-delta 10 --lr-schedule cosine --ema-decay 0.999 --stream \
    --throttle-file logs/voice-three.throttle.json --save-every 1000 --keep-every 1000 --max-seconds 172800 >> logs/voice-three.log 2>&1
  kill $G 2>/dev/null
  log "voice-three done"
fi
