#!/bin/bash
# Voice block 2: emotion input (7 labels x intensity), 12 voices = 6 recorded (IT1 + 5 CML) + 6 synthetic emotional
# (Qwen3-TTS VoiceDesign). Branch of runs/voice at 64168, +50k steps, no GAN, vocoder-input loss. MODE=resume to continue.
set -e
cd "$(dirname "$0")/../.."
grep -q EMOTION-PREP-DONE logs/emotion-prep.log || { echo "emotion data not ready"; exit 3; }
.venv/bin/python logs/label-emotions.py
START=(--init-from runs/voice/last.pt --new-data)
[ "$MODE" = resume ] && START=(--resume runs/voice-emotion/last.pt)
.venv/bin/python scripts/gpu_governor.py --file logs/voice.throttle.json --max-util 85 --target-util 80 --max-power 250 --start 0 \
    >> logs/voice.governor.log 2>&1 &
GOVERNOR=$!
trap 'kill $GOVERNOR 2>/dev/null || true' EXIT
.venv/bin/python -m tiny_voca.train "${START[@]}" \
  --train data/it-multi6-annotations/train.jsonl data/emotion-annotations/train.jsonl \
  --validation data/it-multi6-annotations/validation.jsonl data/emotion-annotations/validation.jsonl --validation-size 84 \
  --output runs/voice-emotion --config configs/italian-voice-emotion.json --steps 114168 --batch-frames 9000 --max-batch 32 \
  --learning-rate 2e-4 --warmup 500 --gan-start 100000000 --teacher-hold 0 --teacher-ramp 1 --teacher-final 0 \
  --weights '{"pitch_spread": 4.0}' --lr-schedule cosine --ema-decay 0.999 --stream \
  --throttle-file logs/voice.throttle.json --save-every 1000 --keep-every 5000 --stop-file logs/voice.stop --max-seconds 172800
echo "VOICE-EXIT $?"
