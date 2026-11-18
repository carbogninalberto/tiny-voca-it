#!/bin/bash
# Pilot: voice model 103k + a direct 100-band Vocos-mel output (vocos_mels=100), 5k steps, same data. GPU 85% / 250 W.
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/gpu_governor.py --file logs/voice-mel-pilot.throttle.json --max-util 85 --target-util 80 --max-power 250 --start 0 \
    >> logs/voice-mel-pilot.governor.log 2>&1 &
GOVERNOR=$!
trap 'kill $GOVERNOR 2>/dev/null || true' EXIT
.venv/bin/python -m tiny_voca.train --init-from runs/voice-emotion/last.pt \
  --train data/it-multi6-annotations/train.jsonl data/emotion-annotations/train.jsonl \
  --validation data/it-multi6-annotations/validation.jsonl data/emotion-annotations/validation.jsonl --validation-size 84 \
  --output runs/voice-mel-pilot --config configs/italian-voice-emotion-mel.json --steps 108292 --batch-frames 9000 --max-batch 32 \
  --learning-rate 1e-4 --warmup 200 --gan-start 100000000 --teacher-hold 0 --teacher-ramp 1 --teacher-final 0 \
  --weights '{"pitch_spread": 4.0}' --lr-schedule cosine --ema-decay 0.999 --stream \
  --throttle-file logs/voice-mel-pilot.throttle.json --save-every 1000 --keep-every 5000 --max-seconds 172800
echo "PILOT-EXIT $?"
