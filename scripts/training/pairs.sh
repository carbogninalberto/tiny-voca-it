#!/bin/bash
# Vocoder fine-tuning pairs from the FINAL emotion voice model (CPU, low priority; the GPU runs the distillation).
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=
nice -n 15 .venv/bin/python -m tiny_voca.vocoder_pairs --checkpoint runs/voice-emotion/inference.pt \
  --annotations data/it-multi6-annotations/train.jsonl data/emotion-annotations/train.jsonl --output data/vocoder-pairs/train
