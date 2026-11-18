#!/bin/bash
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= PYTHONPATH=$EXTRA_LIBS_DIR/pylib
exec .venv/bin/python scripts/playground_server.py --model runs/voice-emotion/inference.pt runs/voice/inference.pt runs/it1-full/inference.pt --vocoder runs/vocoder-prior/vocoder.pt runs/vocoder-distill/vocoder.pt --port 8790
