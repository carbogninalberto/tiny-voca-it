#!/usr/bin/env bash
# Round 6: vocoder fine-tune with the high-band pulse share drawn per utterance from [0, 0.1],
# so per-voice settings (IT1 0.0, EF3 0.05) are in distribution. Resumable.
cd "$(dirname "$0")/.."
LIB=${EXTRA_LIBS_DIR:-$HOME/.local/share/tiny-voca-libs}
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/round6.log; }
governor(){ .venv/bin/python scripts/gpu_governor.py --file $1 --max-util 85 --target-util 80 --max-power 250 --start 0 >> $1.log 2>&1 & echo $!; }
if ! grep -q '"done"' logs/vocoder-three.log 2>/dev/null; then
  START=(--init-from runs/vocoder-two/last.pt --fresh-schedule); [ -f runs/vocoder-three/last.pt ] && START=(--resume)
  log "vocoder-three: start"; G=$(governor logs/vocoder-three.throttle.json)
  PYTHONPATH=$LIB/pylib .venv/bin/python -m tiny_voca.train_vocoder "${START[@]}" --prior --prior-high-hz 2500 --prior-high-gain 0.05 \
    --prior-gain-range 0.0 0.1 \
    --manifests data/it-multi-features/train-it1.jsonl data/ef3-features/train.jsonl \
    --validation data/it1-full-features/test.jsonl --output runs/vocoder-three --steps 210000 --batch 64 --learning-rate 5e-5 --warmup 300 --mel-weight 15 \
    --teacher vocos --distill-mode spectrum --distill-weight 4 --save-every 1000 --keep-every 2000 --throttle-file logs/vocoder-three.throttle.json >> logs/vocoder-three.log 2>&1
  kill $G 2>/dev/null
  grep -q '"done"' logs/vocoder-three.log || { log "vocoder-three failed"; exit 1; }
fi
log "round 6 done"
