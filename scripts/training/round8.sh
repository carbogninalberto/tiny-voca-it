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
