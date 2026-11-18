#!/bin/bash
# Synthetic Italian emotion data: 6 designed voices x (neutral + 6 emotions x 2 strengths) x 30 sentences, VoiceDesign direct.
cd "$(dirname "$0")/../.."
PYTHONPATH=$EXTRA_LIBS_DIR/qwenlib .venv/bin/python scripts/generate_emotions.py --root data/emotion-synth --clips-per-reference 30 --batch 8
echo GEN-EXIT $?
