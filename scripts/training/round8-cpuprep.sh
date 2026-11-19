#!/usr/bin/env bash
# CPU pipelining for round 8: features and alignment of each finished voice while the GPU generates the
# next one. Same arguments as round 8 stage 3, so that stage's --resume skips the finished work.
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=
log(){ echo "$(date '+%m-%d %H:%M') $*" >> logs/round8.log; }
prep(){ local set=$1
  for s in validation test train; do
    [ -s data/$set-synth/$s.jsonl ] || continue
    nice -n 10 .venv/bin/python -m tiny_voca.prepare --inputs data/$set-synth/$s.jsonl --split $s --output data/$set-features \
      --config configs/italian-voice-round-8.json --workers 8 --resume
  done
  nice -n 10 .venv/bin/python -m tiny_voca.annotate --features data/$set-features --output data/$set-annotations \
    --dataset-revision qwen3-$set-20261119 --config configs/italian-voice-round-8.json --device cpu --resume
}
for V in EM4 EF4; do
  set=$(echo $V | tr A-Z a-z)
  until grep -q "$V neutral generation done" logs/round8.log; do sleep 60; done
  log "cpu prep $set: start"
  ( set -e; prep $set; echo "$set-CPUPREP-DONE" ) >> logs/round8-cpuprep.log 2>&1
  grep -q "$set-CPUPREP-DONE" logs/round8-cpuprep.log && log "cpu prep $set: done" || log "cpu prep $set: failed (stage 3 will redo it)"
done
