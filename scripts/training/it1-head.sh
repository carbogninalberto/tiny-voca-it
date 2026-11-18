#!/bin/bash
# Neural spectral head test: branch of it1-full at step 14000, 10k steps, GPU capped ~80%.
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/gpu_governor.py --file logs/it1-head.throttle.json --max-util 80 --target-util 75 --max-power 300 --start 0 \
    > logs/it1-head.governor.log 2>&1 &
GOVERNOR=$!
trap 'kill $GOVERNOR 2>/dev/null || true' EXIT
.venv/bin/python -m tiny_voca.train --init-from $EXTRA_LIBS_DIR/it1-full-branchpoint.pt \
  --train data/it1-full-annotations/train.jsonl --validation data/it1-full-annotations/validation.jsonl \
  --output runs/it1-head --config configs/italian-it1-head.json --steps 24000 --batch-frames 9000 --max-batch 32 \
  --learning-rate 3e-4 --warmup 500 --gan-start 0 --gan-ramp 1000 --gan-ratio 0.7 --teacher-hold 0 --teacher-ramp 1 --teacher-final 0 \
  --lr-schedule cosine --ema-decay 0.999 --stream --throttle-file logs/it1-head.throttle.json --save-every 1000 --keep-every 2000
echo HEAD-DONE
