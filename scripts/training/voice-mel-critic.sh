#!/bin/bash
# Round 4 voice: adversarial 113k -> 128k with the mel critic and a stronger adversarial share (no pulse gate).
cd "$(dirname "$0")/../.."
export PYTHONPATH=$EXTRA_LIBS_DIR
.venv/bin/python scripts/gpu_governor.py --file logs/voice-mel.throttle.json --max-util 85 --target-util 80 --max-power 250 --start 0 >> logs/voice-mel.throttle.json.log 2>&1 &
G=$!
START=(--init-from runs/voice-adv-pilot/last.pt); [ -f runs/voice-mel-critic/last.pt ] && START=(--resume runs/voice-mel-critic/last.pt)
.venv/bin/python -m tiny_voca.train "${START[@]}" \
  --train data/it-multi6-annotations/train.jsonl data/emotion-annotations/train.jsonl \
  --validation data/it-multi6-annotations/validation.jsonl data/emotion-annotations/validation.jsonl --validation-size 84 \
  --output runs/voice-mel-critic --config configs/italian-voice-emotion.json --steps 128292 --batch-frames 9000 --max-batch 32 \
  --learning-rate 1e-4 --warmup 200 --gan-start 0 --gan-ramp 1000 --gan-ratio 1.0 --mel-critic --teacher-hold 0 --teacher-ramp 1 --teacher-final 0 \
  --weights '{"pitch_spread": 4.0}' --lr-schedule cosine --ema-decay 0.999 --stream \
  --throttle-file logs/voice-mel.throttle.json --save-every 1000 --keep-every 2500 --max-seconds 172800
echo "VOICE-MEL-EXIT $?"
kill $G 2>/dev/null
