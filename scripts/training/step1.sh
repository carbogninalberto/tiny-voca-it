#!/bin/bash
# Step 1: all 23.5 h of IT1 (validation/test identical to it-v2), cosine LR + EMA, GPU capped ~80%.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=~/.local/bin:$PATH
PY=.venv/bin/python
for split in validation test train; do
  $PY -m tiny_voca.prepare --inputs data/it-1595-full/$split.jsonl --split $split --output data/it1-full-features \
      --config configs/italian-it1.json --workers 11 --reuse data/it-features --resume
done
$PY -m tiny_voca.annotate --features data/it1-full-features --output data/it1-full-annotations \
    --dataset-revision cml-tts-48aedd7a-speaker1595-full --config configs/italian-it1.json --resume
$PY scripts/gpu_governor.py --file logs/it1-full.throttle.json --max-util 80 --target-util 75 --max-power 240 \
    > logs/it1-full.governor.log 2>&1 &
GOVERNOR=$!
trap 'kill $GOVERNOR 2>/dev/null || true' EXIT
$PY -m tiny_voca.train --train data/it1-full-annotations/train.jsonl --validation data/it1-full-annotations/validation.jsonl \
    --output runs/it1-full --config configs/italian-it1.json --steps 100000 --batch-frames 9000 --max-batch 32 \
    --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 --lr-schedule cosine --ema-decay 0.999 \
    --stream --throttle-file logs/it1-full.throttle.json --max-seconds 172800 --save-every 1000 --keep-every 10000
echo STEP1-DONE
