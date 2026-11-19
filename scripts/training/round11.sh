#!/usr/bin/env bash
# Round 11: tiny-voca-it-zs pilot data, 200 designed voices x 24 cloned sentences (20 voices held out). Resumable.
cd "$(dirname "$0")/.."
LIB=${EXTRA_LIBS_DIR:-$HOME/.local/share/tiny-voca-libs}
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/round11.log; }
if ! grep -q '"complete": true' logs/zs-pool.log 2>/dev/null; then
  log "zs voice pool: start"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$LIB/qwenlib .venv/bin/python scripts/generate_voice_pool.py \
    --root data/zs-pilot --voices 200 --clips 24 --held-out 20 --batch 16 --memory-fraction 0.8 >> logs/zs-pool.log 2>&1
  grep -q '"complete": true' logs/zs-pool.log || { log "zs voice pool failed"; exit 1; }
fi
log "zs voice pool done: $(grep '"complete": true' logs/zs-pool.log | tail -1)"
log "round 11 done"
