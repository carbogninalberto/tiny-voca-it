#!/bin/bash
# Vocoder fine-tune on the voice model's own spectrograms (input) -> real recordings (target), plus distillation from
# Vocos on the REAL spectrogram (clean target). Warm start from runs/vocoder-distill (160k) -> 190k. GPU 85% / 250 W.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$EXTRA_LIBS_DIR/pylib
START=(--init-from runs/vocoder-distill/last.pt)
[ -f runs/vocoder-ft/last.pt ] && START=(--resume)
.venv/bin/python scripts/gpu_governor.py --file logs/vocoder-ft.throttle.json --max-util 85 --target-util 80 --max-power 250 --start 0.02 \
    >> logs/vocoder-ft.governor.log 2>&1 &
GOVERNOR=$!
trap 'kill $GOVERNOR 2>/dev/null || true' EXIT
.venv/bin/python -m tiny_voca.train_vocoder "${START[@]}" --manifests data/it-multi-features/train.jsonl \
  --pairs data/vocoder-pairs/train/pairs.jsonl --pair-fraction 0.7 \
  --validation data/it1-full-features/test.jsonl --output runs/vocoder-ft --steps 190000 --batch 64 \
  --teacher vocos --distill-weight 10 --keep-every 10000 --throttle-file logs/vocoder-ft.throttle.json
