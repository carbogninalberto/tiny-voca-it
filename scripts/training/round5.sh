#!/bin/bash
# Round 5: two voices (IT1 male + EF3 female). One GPU job at a time; every stage resumable / skipped when done.
#   1. after round 4 is scored: ~6 h of EF3 speech, Qwen3-TTS Base cloning one reference (EF3-option-1)
#   2. features + alignment for it (GPU)
#   3. waits for logs/round5-base.txt (the voice checkpoint to start from, chosen from round 4's scores),
#      then fine-tunes on IT1 + EF3 + the emotion set (without the VoiceDesign EF3 neutral clips, whose
#      timbre drifts), 20k steps, adversarial ratio 0.7 (the recipe that helped; no mel critic)
#   4. vocoder 180k -> 200k on IT1 + EF3, high-band prior (2.5 kHz, gain 0.05), mel weight 15 (was 45)
cd "$(dirname "$0")/../.."
LIB=${EXTRA_LIBS_DIR:-$HOME/.local/share/tiny-voca-libs}
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/round5.log; }
step_of(){ .venv/bin/python -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=True)['step'])" "$1" 2>/dev/null || echo 0; }
governor(){ .venv/bin/python scripts/gpu_governor.py --file $1 --max-util 85 --target-util 80 --max-power 250 --start 0 >> $1.log 2>&1 & echo $!; }

if ! grep -q '"complete": true' logs/ef3-gen.log 2>/dev/null; then
  log "EF3 generation: start"
  # Generation is bound by one CPU core driving small GPU kernels: batch time is nearly flat from 4 to 16
  # (benchmark: 1.6x real time at batch 8, 3.1x at 16, 3.3x at 24), and parallel processes only time-slice
  # the GPU. So: one process, a large batch, Whisper screening on idle CPU cores overlapped with generation.
  # Fallback on failure (e.g. out of GPU memory): a smaller batch; the run resumes where it stopped.
  for B in 18 12 8; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$LIB/qwenlib .venv/bin/python scripts/generate_voice.py \
      --root data/ef3-synth --voice-id EF3 --reference data/emotion-synth/audio/emo-EF3-neutral-0.0-3e06518d08.wav \
      --hours 6 --batch $B --whisper-device cuda --memory-fraction 0.8 >> logs/ef3-gen.log 2>&1 && break
    log "EF3 generation failed at batch $B"
  done
  PYTHONPATH=$LIB/qwenlib .venv/bin/python scripts/generate_voice.py --root data/ef3-synth --voice-id EF3 \
    --reference data/emotion-synth/audio/emo-EF3-neutral-0.0-3e06518d08.wav --hours 0 >> logs/ef3-gen.log 2>&1
  grep -q '"complete": true' logs/ef3-gen.log || { log "EF3 generation failed"; exit 1; }
fi
log "EF3 generation done: $(grep -o '"accepted_hours": [0-9.]*' logs/ef3-gen.log | tail -1)"

if ! grep -q "EF3-PREP-DONE" logs/ef3-prep.log 2>/dev/null; then
  log "EF3 prep: start"
  ( set -e
    for s in validation test train; do
      .venv/bin/python -m tiny_voca.prepare --inputs data/ef3-synth/$s.jsonl --split $s --output data/ef3-features \
        --config configs/italian-voice-emotion.json --workers 10 --resume
    done
    .venv/bin/python -m tiny_voca.annotate --features data/ef3-features --output data/ef3-annotations \
      --dataset-revision qwen3-clone-ef3-20261118 --config configs/italian-voice-emotion.json --device cuda --resume
    echo EF3-PREP-DONE ) >> logs/ef3-prep.log 2>&1
  grep -q "EF3-PREP-DONE" logs/ef3-prep.log || { log "EF3 prep failed"; exit 1; }
fi
log "EF3 prep done"

until [ -f logs/round5-base.txt ]; do sleep 60; done
BASE=$(cat logs/round5-base.txt); TARGET=$(( $(step_of $BASE) + 20000 ))
if [ "$(step_of runs/voice-two/last.pt)" -lt "$TARGET" ] || [ ! -f runs/voice-two/last.pt ]; then
  START=(--init-from $BASE --new-data); [ -f runs/voice-two/last.pt ] && START=(--resume runs/voice-two/last.pt)
  log "voice-two: start (${START[*]}) to $TARGET"; G=$(governor logs/voice-two.throttle.json)
  PYTHONPATH=$LIB/pylib .venv/bin/python -m tiny_voca.train "${START[@]}" \
    --train data/it-multi6-annotations/train-it1.jsonl data/ef3-annotations/train.jsonl data/emotion-annotations/train-no-ef3-neutral.jsonl \
    --validation data/it-multi6-annotations/validation-it1.jsonl data/ef3-annotations/validation.jsonl data/emotion-annotations/validation.jsonl \
    --validation-size 96 --output runs/voice-two --config configs/italian-voice-emotion.json --steps $TARGET --batch-frames 9000 --max-batch 32 \
    --learning-rate 1e-4 --warmup 300 --gan-start 0 --gan-ramp 1000 --gan-ratio 0.7 --teacher-hold 0 --teacher-ramp 1 --teacher-final 0 \
    --weights '{"pitch_spread": 4.0}' --lr-schedule cosine --ema-decay 0.999 --stream \
    --throttle-file logs/voice-two.throttle.json --save-every 1000 --keep-every 5000 --max-seconds 172800 >> logs/voice-two.log 2>&1
  kill $G 2>/dev/null
  [ "$(step_of runs/voice-two/last.pt)" -ge "$TARGET" ] || { log "voice-two stopped early at $(step_of runs/voice-two/last.pt)"; exit 1; }
fi
log "voice-two done"

if ! grep -q '"done"' logs/vocoder-two.log 2>/dev/null; then
  START=(--init-from runs/vocoder-prior2/last.pt --fresh-schedule); [ -f runs/vocoder-two/last.pt ] && START=(--resume)
  log "vocoder-two: start"; G=$(governor logs/vocoder-two.throttle.json)
  PYTHONPATH=$LIB/pylib .venv/bin/python -m tiny_voca.train_vocoder "${START[@]}" --prior --prior-high-hz 2500 --prior-high-gain 0.05 \
    --manifests data/it-multi-features/train-it1.jsonl data/ef3-features/train.jsonl \
    --validation data/it1-full-features/test.jsonl --output runs/vocoder-two --steps 200000 --batch 64 --learning-rate 1e-4 --warmup 500 --mel-weight 15 \
    --teacher vocos --distill-mode spectrum --distill-weight 4 --keep-every 5000 --throttle-file logs/vocoder-two.throttle.json >> logs/vocoder-two.log 2>&1
  kill $G 2>/dev/null
  grep -q '"done"' logs/vocoder-two.log || { log "vocoder-two stopped early"; exit 1; }
fi
log "round 5 done"
