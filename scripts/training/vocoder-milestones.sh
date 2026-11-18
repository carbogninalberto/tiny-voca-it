#!/bin/bash
# Keep full state + inference weights of the vocoder run at fixed steps; waits through pauses (saves every 2000 steps).
cd "$(dirname "$0")/../.."
step(){ .venv/bin/python -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=True)['step'])" "$1" 2>/dev/null || echo 0; }
for target in 50000 80000 100000 150000 200000 250000 300000; do
  name=$(printf "%07d" $target)
  until [ -f runs/vocoder/keep-$name/last.pt ]; do
    if [ "$(step runs/vocoder/last.pt)" = "$target" ]; then
      mkdir -p runs/vocoder/keep-$name.tmp && cp runs/vocoder/last.pt runs/vocoder/vocoder.pt runs/vocoder/keep-$name.tmp/
      if [ "$(step runs/vocoder/keep-$name.tmp/last.pt)" = "$target" ] && [ "$(step runs/vocoder/keep-$name.tmp/vocoder.pt)" = "$target" ]; then
        mv runs/vocoder/keep-$name.tmp runs/vocoder/keep-$name; echo "$(date +%H:%M) kept $target"
      else rm -rf runs/vocoder/keep-$name.tmp; fi
    fi
    [ "$(step runs/vocoder/last.pt)" -gt "$target" ] && { echo "$(date +%H:%M) missed $target"; break; }
    sleep 20
  done
done
