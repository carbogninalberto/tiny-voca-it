#!/usr/bin/env bash
# Round 12 (zero-shot A2): real Italian speakers (VoxPopuli) + the 180 synthetic voices, fingerprint noise,
# fine-tuned from the pilot; judged on held-out REAL speakers. Resumable.
cd "$(dirname "$0")/.."
LIB=${EXTRA_LIBS_DIR:-$HOME/.local/share/tiny-voca-libs}
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/round12.log; }
if ! grep -q 'zs-real import done' logs/round12.log 2>/dev/null; then
  log "zs-real import: start"
  nice -n 5 .venv/bin/python scripts/import_voxpopuli.py --source data/voxpopuli-it/it --root data/zs-real --clips 25 --held-out 40 >> logs/zs-real.log 2>&1 \
    || { log "zs-real import failed"; exit 1; }
  log "zs-real import done: $(tail -1 logs/zs-real.log)"
fi
if ! grep -q 'zs-real prep done' logs/round12.log; then
  log "zs-real prep: start"
  ( set -e
    .venv/bin/python scripts/voice_fingerprints.py --root data/zs-real --output data/zs-real/fingerprints.pt
    .venv/bin/python - <<'PY'
import json, torch
a = torch.load("data/zs-pilot/fingerprints.pt", weights_only=False); b = torch.load("data/zs-real/fingerprints.pt", weights_only=False)
torch.save({"fingerprints": {**a["fingerprints"], **b["fingerprints"]}, "held_out_clips": {**a["held_out_clips"], **b["held_out_clips"]},
            "model": a["model"]}, "data/zs-real/fingerprints-all.pt")
c = json.load(open("configs/zs-pilot.json")); voices = json.load(open("data/zs-real/voices.json"))
c["speakers"] = list(c["speakers"]) + sorted(voices); json.dump(c, open("configs/zs-real.json", "w"), indent=2, ensure_ascii=False)
print("speakers", len(c["speakers"]))
PY
    for s in validation test train; do
      .venv/bin/python -m tiny_voca.prepare --inputs data/zs-real/$s.jsonl --split $s --output data/zs-real-features \
        --config configs/zs-real.json --workers 8 --resume
    done
    .venv/bin/python -m tiny_voca.annotate --features data/zs-real-features --output data/zs-real-annotations \
      --dataset-revision voxpopuli-it-20261119 --config configs/zs-real.json --device cuda --resume
    echo ZS-REAL-PREP-DONE ) >> logs/zs-real-prep.log 2>&1
  grep -q ZS-REAL-PREP-DONE logs/zs-real-prep.log || { log "zs-real prep failed"; exit 1; }
  log "zs-real prep done"
fi
if ! grep -q 'zs-real train done' logs/round12.log; then
  START=(--init-from runs/zs-pilot/last-ema.pt --new-data); [ -f runs/zs-real/last.pt ] && START=(--resume)
  log "zs-real train: start"
  PYTHONPATH=$LIB/pylib .venv/bin/python -m tiny_voca.train "${START[@]}" --speaker-fingerprints data/zs-real/fingerprints-all.pt --fingerprint-noise 0.03 \
    --train data/zs-annotations/train.jsonl data/zs-real-annotations/train.jsonl \
    --validation data/zs-annotations/validation.jsonl data/zs-real-annotations/validation.jsonl --validation-size 96 \
    --output runs/zs-real --config configs/zs-real.json --steps 169000 --batch-frames 9000 --max-batch 32 \
    --learning-rate 1e-4 --warmup 300 --gan-start 0 --gan-ramp 1000 --gan-ratio 0.7 --teacher-hold 0 --teacher-ramp 1 --teacher-final 0 \
    --weights '{"pitch_spread": 4.0}' --pitch-delta 10 --texture 20 --lr-schedule cosine --ema-decay 0.999 --stream \
    --save-every 1000 --keep-every 5000 --max-seconds 172800 >> logs/zs-real-train.log 2>&1
  [ -f runs/zs-real/inference.pt ] && grep -q '"step": 169000' runs/zs-real/metrics.jsonl 2>/dev/null || { log "zs-real train failed"; exit 1; }
  log "zs-real train done"
fi
if ! grep -q 'zs-real eval done' logs/round12.log; then
  for root in zs-real zs-pilot; do
    .venv/bin/python scripts/zs_eval.py --checkpoint runs/zs-real/inference.pt --vocoder runs/vocoder-four/vocoder-voicegain.pt \
      --root data/$root --fingerprints data/zs-real/fingerprints-all.pt --save runs/zs-real/samples-$root >> logs/zs-real-eval.log 2>&1
  done
  log "zs-real eval done: $(grep -E '^unseen' logs/zs-real-eval.log | tr '\n' ' ')"
fi
log "round 12 done"
