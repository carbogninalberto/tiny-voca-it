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
log "zs pool stage complete"

# Stage A of the zero-shot pilot: fingerprints, features, fine-tune with fingerprint voices, similarity test.
if ! grep -q 'zs prep done' logs/round11.log; then
  log "zs prep: start"
  ( set -e
    .venv/bin/python scripts/voice_fingerprints.py --root data/zs-pilot --output data/zs-pilot/fingerprints.pt
    for s in validation test train; do
      .venv/bin/python -m tiny_voca.prepare --inputs data/zs-pilot/$s.jsonl --split $s --output data/zs-features \
        --config configs/zs-pilot.json --workers 10 --resume
    done
    .venv/bin/python -m tiny_voca.annotate --features data/zs-features --output data/zs-annotations \
      --dataset-revision qwen3-zs-pilot-20261119 --config configs/zs-pilot.json --device cuda --resume
    echo ZS-PREP-DONE ) >> logs/zs-prep.log 2>&1
  grep -q ZS-PREP-DONE logs/zs-prep.log || { log "zs prep failed"; exit 1; }
  log "zs prep done"
fi
if ! grep -q 'zs train done' logs/round11.log; then
  START=(--init-from runs/voice-six/step-00139000-ema.pt --new-data); [ -f runs/zs-pilot/last.pt ] && START=(--resume)
  log "zs train: start"
  PYTHONPATH=$LIB/pylib .venv/bin/python -m tiny_voca.train "${START[@]}" --speaker-fingerprints data/zs-pilot/fingerprints.pt \
    --train data/zs-annotations/train.jsonl --validation data/zs-annotations/validation.jsonl --validation-size 96 \
    --output runs/zs-pilot --config configs/zs-pilot.json --steps 154000 --batch-frames 9000 --max-batch 32 \
    --learning-rate 1e-4 --warmup 300 --gan-start 0 --gan-ramp 1000 --gan-ratio 0.7 --teacher-hold 0 --teacher-ramp 1 --teacher-final 0 \
    --weights '{"pitch_spread": 4.0}' --pitch-delta 10 --texture 20 --lr-schedule cosine --ema-decay 0.999 --stream \
    --save-every 1000 --keep-every 5000 --max-seconds 172800 >> logs/zs-train.log 2>&1
  [ -f runs/zs-pilot/inference.pt ] && grep -q '"step": 154000' runs/zs-pilot/metrics.jsonl 2>/dev/null || { log "zs train failed"; exit 1; }
  log "zs train done"
fi
if ! grep -q 'zs eval done' logs/round11.log; then
  .venv/bin/python scripts/zs_eval.py --checkpoint runs/zs-pilot/inference.pt --vocoder runs/vocoder-four/vocoder-voicegain.pt \
    --root data/zs-pilot --fingerprints data/zs-pilot/fingerprints.pt --save runs/zs-pilot/samples >> logs/zs-eval.log 2>&1 \
    && log "zs eval done: $(tail -1 logs/zs-eval.log)" || log "zs eval failed"
fi
log "zs pilot stage A done"
log "round 11 done"
