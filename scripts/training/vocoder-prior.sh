#!/bin/bash
# Harmonic-prior vocoder (2.22M): phase from a pulse train built from the pitch, network predicts magnitudes and a
# phase correction. Warm start from the distilled vocoder; trains on real spectrograms AND the voice model's own
# spectrograms (pairs, 50%), magnitude distillation from Vocos on the real spectrogram. +40k steps. GPU 85% / 250 W.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$EXTRA_LIBS_DIR/pylib
START=(--init-from runs/vocoder-distill/last.pt)
[ -f runs/vocoder-prior/last.pt ] && START=(--resume)
.venv/bin/python scripts/gpu_governor.py --file logs/vocoder-prior.throttle.json --max-util 85 --target-util 80 --max-power 250 --start 0.02 \
    >> logs/vocoder-prior.governor.log 2>&1 &
GOVERNOR=$!
trap 'kill $GOVERNOR 2>/dev/null || true' EXIT
.venv/bin/python -m tiny_voca.train_vocoder "${START[@]}" --prior --manifests data/it-multi-features/train.jsonl \
  --pairs data/vocoder-pairs/train/pairs.jsonl --pair-fraction 0.5 \
  --validation data/it1-full-features/test.jsonl --output runs/vocoder-prior --steps 184000 --batch 64 \
  --teacher vocos --distill-mode spectrum --distill-weight 4 --keep-every 10000 --throttle-file logs/vocoder-prior.throttle.json \
  --fresh-schedule --learning-rate 2e-4 --warmup 1000
