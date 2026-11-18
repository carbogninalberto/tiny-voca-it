#!/bin/bash
# Features + phoneme alignment for the synthetic emotion set (CPU only; the GPU belongs to the running block).
set -e
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=
for s in validation test train; do
  nice -n 15 .venv/bin/python -m tiny_voca.prepare --inputs data/emotion-synth/$s.jsonl --split $s --output data/emotion-features \
    --config configs/italian-voice-emotion.json --workers 4 --resume
done
nice -n 15 .venv/bin/python -m tiny_voca.annotate --features data/emotion-features --output data/emotion-annotations \
  --dataset-revision qwen3-voicedesign-emotion-20261117 --config configs/italian-voice-emotion.json --device cpu --resume
echo EMOTION-PREP-DONE
