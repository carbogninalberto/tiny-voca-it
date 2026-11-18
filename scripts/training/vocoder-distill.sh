#!/bin/bash
# Distilled vocoder: same 2.1M model, warm start from runs/vocoder at 100k, trained with the usual losses plus a
# frame-level match of the pretrained Vocos spectrum (magnitude + phase) (teacher, training only). 100k -> 160k, batch 64, GPU 85% / 250 W.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$EXTRA_LIBS_DIR/pylib
START=(--init-from runs/vocoder/keep-0100000/last.pt)
[ -f runs/vocoder-distill/last.pt ] && START=(--resume)
.venv/bin/python scripts/gpu_governor.py --file logs/vocoder-distill.throttle.json --max-util 85 --target-util 80 --max-power 250 --start 0.02 \
    >> logs/vocoder-distill.governor.log 2>&1 &
GOVERNOR=$!
trap 'kill $GOVERNOR 2>/dev/null || true' EXIT
.venv/bin/python -m tiny_voca.train_vocoder "${START[@]}" --manifests data/it-multi-features/train.jsonl \
  --exclude-speakers CML659 CML10446 CML2019 CML1725 --validation data/it1-full-features/test.jsonl \
  --output runs/vocoder-distill --steps 160000 --batch 64 --teacher vocos --distill-mode spectrum --distill-weight 4 \
  --keep-every 20000 --throttle-file logs/vocoder-distill.throttle.json
