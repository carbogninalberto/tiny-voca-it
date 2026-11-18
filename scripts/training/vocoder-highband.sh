#!/bin/bash
# Round 4 vocoder: 180k -> 195k with the prior's pulses weakened above 2.5 kHz (gain 0.05 from 5 kHz).
# Starts after the round-4 voice run and its scoring (one GPU job at a time).
cd "$(dirname "$0")/../.."
export PYTHONPATH=$EXTRA_LIBS_DIR
until grep -q "VOICE-MEL-EXIT" logs/voice-mel-critic.log && [ -f /tmp/claude-1000/-home-alberto-repos-tiny-voca-it/98a6e7b5-7487-4cb9-b31f-c6107f43bce2/scratchpad/mel-critic-key.json ]; do sleep 60; done
.venv/bin/python scripts/gpu_governor.py --file logs/vocoder-highband.throttle.json --max-util 85 --target-util 80 --max-power 250 --start 0 >> logs/vocoder-highband.throttle.json.log 2>&1 &
G=$!
START=(--init-from runs/vocoder-prior2/last.pt --fresh-schedule); [ -f runs/vocoder-highband/last.pt ] && START=(--resume)
.venv/bin/python -m tiny_voca.train_vocoder "${START[@]}" --prior --prior-high-hz 2500 --prior-high-gain 0.05 \
  --manifests data/it-multi-features/train.jsonl --pairs data/vocoder-pairs-gate/train/pairs.jsonl --pair-fraction 0.5 \
  --validation data/it1-full-features/test.jsonl --output runs/vocoder-highband --steps 195000 --batch 64 --learning-rate 1e-4 --warmup 500 \
  --teacher vocos --distill-mode spectrum --distill-weight 4 --keep-every 5000 --throttle-file logs/vocoder-highband.throttle.json
echo "VOCODER-HB-EXIT $?"
kill $G 2>/dev/null
