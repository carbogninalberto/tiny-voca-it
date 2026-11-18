#!/bin/bash
cd /home/alberto/repos/tiny-voca-it
export CUDA_VISIBLE_DEVICES= PYTHONPATH=/tmp/claude-1000/-home-alberto-repos-tiny-voca-it/98a6e7b5-7487-4cb9-b31f-c6107f43bce2/scratchpad/pylib
exec .venv/bin/python scripts/playground_server.py --model runs/voice-two/inference-115000.pt runs/voice-two/inference.pt runs/voice-adv-pilot/inference.pt runs/voice-emotion/inference.pt runs/voice/inference.pt runs/it1-full/inference.pt --vocoder runs/vocoder-two/vocoder-voicegain.pt runs/vocoder-two/vocoder-0200000.pt runs/vocoder-prior2/vocoder.pt runs/vocoder-prior/vocoder-0150000.pt runs/vocoder-distill/vocoder.pt --port 8790 --voices IT1 EF3
