#!/bin/bash
cd "$(dirname "$0")/../.."
PYTHONPATH=$EXTRA_LIBS_DIR/qwenlib .venv/bin/python scripts/generate_emotions.py --root data/emotion-pilot --pilot
echo PILOT-EXIT $?
