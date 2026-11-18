#!/bin/bash
# Phase 2: small Italian neural vocoder (~2M params), 500k steps, GPU capped ~80%. Resumable: rerun with RESUME=1.
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/gpu_governor.py --file logs/vocoder.throttle.json --max-util 85 --target-util 80 --max-power 250 --start 0.02 \
    > logs/vocoder.governor.log 2>&1 &
GOVERNOR=$!
trap 'kill $GOVERNOR 2>/dev/null || true' EXIT
.venv/bin/python -m tiny_voca.train_vocoder --manifests data/it-multi-features/train.jsonl \
  --exclude-speakers CML659 CML10446 CML2019 CML1725 --validation data/it1-full-features/test.jsonl \
  --output runs/vocoder --steps 300000 --batch 64 --throttle-file logs/vocoder.throttle.json ${RESUME:+--resume}
echo VOCODER-DONE
