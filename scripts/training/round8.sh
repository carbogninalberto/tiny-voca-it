#!/usr/bin/env bash
# Round 8: new product voices EM4 (design M4) and EF4 (design F3) + much stronger emotion data.
#  1. neutral data: Qwen3-TTS Base clones each reference on CML-TTS sentences (5 h screened per voice)
#  2. emotion data: VoiceDesign with the voice description + style, on the new emotion texts (~120 per emotion)
#  3. voice-similarity filter (WavLM-SV) against the reference, then splits, features, alignment
#  4. voice fine-tune from voice-two 115k (emotion data oversampled, pitch-change loss), 5. vocoder fine-tune
# Every stage is resumable.
cd "$(dirname "$0")/.."
LIB=${EXTRA_LIBS_DIR:-$HOME/.local/share/tiny-voca-libs}
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/round8.log; }
governor(){ .venv/bin/python scripts/gpu_governor.py --file $1 --max-util 85 --target-util 80 --max-power 250 --start 0 >> $1.log 2>&1 & echo $!; }
TRAIN_TEXTS="data/it-multi6-annotations/train.jsonl data/emotion-annotations/train.jsonl data/ef3-synth/train.jsonl"
VAL_TEXTS="data/it-multi6-annotations/validation.jsonl data/it-multi6-annotations/test.jsonl data/emotion-annotations/validation.jsonl data/ef3-synth/validation.jsonl data/ef3-synth/test.jsonl"

neutral(){  # voice, root, extra train-text manifests, extra validation-text manifests
  local V=$1 R=$2
  if ! grep -q '"complete": true' logs/$V-gen.log 2>/dev/null; then
    log "$V neutral generation: start"
    for B in 18 12 8; do
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$LIB/qwenlib .venv/bin/python scripts/generate_voice.py \
        --root $R --voice-id $V --reference data/voice-refs/$V.wav --reference-manifest data/voice-refs/manifest.jsonl --hours 5 \
        --batch $B --whisper-device cuda --memory-fraction 0.8 \
        --exclude-texts data/it-multi-annotations/test.jsonl data/it-multi6-annotations/test.jsonl data/ef3-synth/test.jsonl $5 \
        --split-train $TRAIN_TEXTS $3 --split-validation $VAL_TEXTS $4 >> logs/$V-gen.log 2>&1 && break
      log "$V generation exited at batch $B; retrying smaller"
    done
    grep -q '"complete": true' logs/$V-gen.log || { log "$V generation failed"; exit 1; }
  fi
  log "$V neutral generation done: $(grep '"complete": true' logs/$V-gen.log | tail -1)"
}
neutral EM4 data/em4-synth "" "" ""
neutral EF4 data/ef4-synth "data/em4-synth/train.jsonl" "data/em4-synth/validation.jsonl data/em4-synth/test.jsonl" "data/em4-synth/test.jsonl"

# Stage 2 waits for the new emotion texts.
until [ -f data/emotion-texts-v2.json ]; do sleep 60; done
if ! grep -q 'EMOTION4-GEN-DONE' logs/emotion4-gen.log 2>/dev/null; then
  log "emotion4 generation: start"
  PYTHONPATH=$LIB/qwenlib .venv/bin/python scripts/generate_emotions.py --root data/emotion4-synth --mode design --voices EM4 EF4 \
    --emotion-texts data/emotion-texts-v2.json --clips-per-reference 100 --batch 16 >> logs/emotion4-gen.log 2>&1 \
    && echo EMOTION4-GEN-DONE >> logs/emotion4-gen.log
  grep -q 'EMOTION4-GEN-DONE' logs/emotion4-gen.log || { log "emotion4 generation failed"; exit 1; }
fi
log "emotion4 generation done"
echo "stages 3-5 are appended once the texts and filter are ready" >> logs/round8.log

# 3. voice-similarity filter + splits for the emotion set, then features and alignment for all three sets.
if ! grep -q 'ROUND8-PREP-DONE' logs/round8-prep.log 2>/dev/null; then
  log "round 8 prep: start"
  ( set -e
    .venv/bin/python scripts/filter_speaker.py --root data/emotion4-synth --references data/voice-refs \
      --split-train $TRAIN_TEXTS data/em4-synth/train.jsonl data/ef4-synth/train.jsonl \
      --split-validation $VAL_TEXTS data/em4-synth/validation.jsonl data/em4-synth/test.jsonl data/ef4-synth/validation.jsonl data/ef4-synth/test.jsonl
    for set in em4 ef4 emotion4; do
      for s in validation test train; do
        [ -s data/$set-synth/$s.jsonl ] || continue
        .venv/bin/python -m tiny_voca.prepare --inputs data/$set-synth/$s.jsonl --split $s --output data/$set-features \
          --config configs/italian-voice-round-8.json --workers 10 --resume
      done
      .venv/bin/python -m tiny_voca.annotate --features data/$set-features --output data/$set-annotations \
        --dataset-revision qwen3-$set-20261119 --config configs/italian-voice-round-8.json --device cuda --resume
    done
    # Emotion clips three times per epoch (oversampling: they are a small share of the hours).
    for k in 1 2 3; do cat data/emotion4-annotations/train.jsonl; done > data/emotion4-annotations/train-x3.jsonl
    echo ROUND8-PREP-DONE ) >> logs/round8-prep.log 2>&1
  grep -q 'ROUND8-PREP-DONE' logs/round8-prep.log || { log "round 8 prep failed"; exit 1; }
fi
log "round 8 prep done"

# 4. voice fine-tune: new voices + emotions (oversampled) + IT1 and the earlier emotion voices for stability.
if ! grep -q 'voice-four done' logs/round8.log; then
  START=(--init-from runs/voice-two/step-00115000.pt --new-data); [ -f runs/voice-four/last.pt ] && START=(--resume)
  log "voice-four: start (${START[*]})"; G=$(governor logs/voice-four.throttle.json)
  PYTHONPATH=$LIB/pylib .venv/bin/python -m tiny_voca.train "${START[@]}" \
    --train data/em4-annotations/train.jsonl data/ef4-annotations/train.jsonl data/emotion4-annotations/train-x3.jsonl \
            data/it-multi6-annotations/train-it1.jsonl data/emotion-annotations/train-no-ef3-neutral.jsonl \
    --validation data/em4-annotations/validation.jsonl data/ef4-annotations/validation.jsonl data/emotion4-annotations/validation.jsonl \
                 data/it-multi6-annotations/validation-it1.jsonl data/emotion-annotations/validation.jsonl \
    --validation-size 96 --output runs/voice-four --config configs/italian-voice-round-8.json --steps 135000 --batch-frames 9000 --max-batch 32 \
    --learning-rate 1e-4 --warmup 300 --gan-start 0 --gan-ramp 1000 --gan-ratio 0.7 --teacher-hold 0 --teacher-ramp 1 --teacher-final 0 \
    --weights '{"pitch_spread": 4.0}' --pitch-delta 10 --new-voice-init EM4=EM1 EF4=EF3 --lr-schedule cosine --ema-decay 0.999 --stream \
    --throttle-file logs/voice-four.throttle.json --save-every 1000 --keep-every 1000 --max-seconds 172800 >> logs/voice-four.log 2>&1
  kill $G 2>/dev/null
  [ -f runs/voice-four/inference.pt ] || { log "voice-four failed"; exit 1; }
  log "voice-four done"
fi

# 5. vocoder fine-tune on the new voices (pulse share drawn from [0, 0.1], per-voice setting chosen afterwards).
if ! grep -q '"done"' logs/vocoder-four.log 2>/dev/null; then
  START=(--init-from runs/vocoder-three/last.pt --fresh-schedule); [ -f runs/vocoder-four/last.pt ] && START=(--resume)
  log "vocoder-four: start"; G=$(governor logs/vocoder-four.throttle.json)
  PYTHONPATH=$LIB/pylib .venv/bin/python -m tiny_voca.train_vocoder "${START[@]}" --prior --prior-high-hz 2500 --prior-high-gain 0.05 \
    --prior-gain-range 0.0 0.1 --manifests data/em4-features/train.jsonl data/ef4-features/train.jsonl data/emotion4-features/train.jsonl \
    data/it-multi-features/train-it1.jsonl --validation data/em4-features/validation.jsonl --output runs/vocoder-four --steps 225000 --batch 64 \
    --learning-rate 5e-5 --warmup 300 --mel-weight 15 --teacher vocos --distill-mode spectrum --distill-weight 4 \
    --save-every 1000 --keep-every 5000 --throttle-file logs/vocoder-four.throttle.json >> logs/vocoder-four.log 2>&1
  kill $G 2>/dev/null
  grep -q '"done"' logs/vocoder-four.log || { log "vocoder-four failed"; exit 1; }
fi
log "round 8 done"
