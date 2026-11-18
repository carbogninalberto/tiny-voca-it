#!/bin/bash
# Voice (acoustic model) for the neural-vocoder pipeline: 6 clean Italian voices (IT1 + 5 CML), branched from
# it1-full step 14000, no GAN, vocoder-input loss, stronger pitch-spread term. 100k voice steps (14000 -> 114000)
# run in blocks: the alternation script stops it with logs/voice.stop. Fresh start, or MODE=resume.
cd "$(dirname "$0")/../.."
START=(--init-from runs/it1-full/last.pt --new-data)
[ "$MODE" = resume ] && START=(--resume runs/voice/last.pt)
.venv/bin/python scripts/gpu_governor.py --file logs/voice.throttle.json --max-util 100 --target-util 97 --max-power 230 --start 0 \
    >> logs/voice.governor.log 2>&1 &
GOVERNOR=$!
trap 'kill $GOVERNOR 2>/dev/null || true' EXIT
.venv/bin/python -m tiny_voca.train "${START[@]}" \
  --train data/it-multi6-annotations/train.jsonl --validation data/it-multi6-annotations/validation.jsonl --validation-size 60 \
  --output runs/voice --config configs/italian-voice.json --steps 114000 --batch-frames 9000 --max-batch 32 \
  --learning-rate 3e-4 --warmup 500 --gan-start 100000000 --teacher-hold 0 --teacher-ramp 1 --teacher-final 0 \
  --weights '{"pitch_spread": 4.0}' --lr-schedule cosine --ema-decay 0.999 --stream \
  --throttle-file logs/voice.throttle.json --save-every 1000 --keep-every 5000 --stop-file logs/voice.stop --max-seconds 172800
echo "VOICE-EXIT $?"
