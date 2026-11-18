#!/bin/bash
# Step 2: 10-speaker base (91 h, IT1 + 9 CML-TTS voices), then full fine-tune on IT1 (23.5 h). GPU capped ~80%.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=~/.local/bin:$PATH
PY=.venv/bin/python
SPEAKERS="4974 4998 6807 2033 1989 2019 659 10446 1725"
until grep -q IT1-IMPORTED logs/run-all.log; do sleep 30; done
sleep 30   # import-multi.log is rewritten by the relaunched import
until grep -q IMPORT-DONE logs/import-multi.log; do
  grep -q Traceback logs/import-multi.log && { echo "import failed"; exit 1; }; sleep 60
done
for split in validation test train; do
  inputs="data/it-1595-full/$split.jsonl"
  for s in $SPEAKERS; do inputs="$inputs data/it-$s-full/$split.jsonl"; done
  $PY -m tiny_voca.prepare --inputs $inputs --split $split --output data/it-multi-features \
      --config configs/italian-multi.json --workers 8 --reuse data/it1-full-features data/it-features
done
$PY -m tiny_voca.annotate --features data/it-multi-features --output data/it-multi-annotations \
    --dataset-revision cml-tts-48aedd7a-multi10 --config configs/italian-multi.json
echo MULTI-ANNOTATED
until grep -q STEP1-DONE logs/step1.log; do sleep 60; done
$PY scripts/gpu_governor.py --file logs/it-multi.throttle.json --max-util 80 --target-util 75 --max-power 300 \
    > logs/it-multi.governor.log 2>&1 &
GOVERNOR=$!
trap 'kill $GOVERNOR 2>/dev/null || true' EXIT
$PY -m tiny_voca.train --train data/it-multi-annotations/train.jsonl --validation data/it-multi-annotations/validation.jsonl \
    --output runs/it-multi --config configs/italian-multi.json --steps 100000 --batch-frames 9000 --max-batch 32 \
    --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 --lr-schedule cosine --ema-decay 0.999 --validation-size 40 \
    --stream --throttle-file logs/it-multi.throttle.json --max-seconds 172800 --save-every 1000 --keep-every 10000
echo BASE-DONE
$PY -m tiny_voca.train --adapt-from runs/it-multi/inference.pt --init-voice IT1 --adapt-mode full \
    --train data/it1-full-annotations/train.jsonl --validation data/it1-full-annotations/validation.jsonl \
    --output runs/it1-from-multi --config configs/italian-it1.json --steps 30000 --batch-frames 9000 --max-batch 32 \
    --learning-rate 1e-4 --warmup 500 --gan-start 1000 --gan-ramp 2000 --teacher-hold 300 --teacher-ramp 700 --teacher-final 0 \
    --lr-schedule cosine --ema-decay 0.999 --stream --throttle-file logs/it-multi.throttle.json \
    --max-seconds 172800 --save-every 1000 --keep-every 5000
echo STEP2-DONE
