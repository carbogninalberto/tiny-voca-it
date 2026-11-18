#!/bin/bash
# Governor for an already running vocoder (PID $1): 85% utilisation, 250 W; stops with it.
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/gpu_governor.py --file logs/vocoder.throttle.json --max-util 85 --target-util 80 --max-power 250 --start 0.1 \
    >> logs/vocoder.governor.log 2>&1 &
G=$!
while kill -0 "$1" 2>/dev/null; do sleep 30; done
kill $G
