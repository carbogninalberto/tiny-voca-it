#!/usr/bin/env bash
# Round 12 (zero-shot A2) data: real Italian speakers. VoxPopuli Italian (CC0) parquet shards.
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY' >> logs/round12-download.log 2>&1
from huggingface_hub import hf_hub_download
for name in ["it/validation-00000-of-00001.parquet"] + [f"it/train-0000{i}-of-00005.parquet" for i in range(5)]:
    print("downloading", name, flush=True)
    hf_hub_download("facebook/voxpopuli", name, repo_type="dataset", local_dir="data/voxpopuli-it")
print("VOXPOPULI-DONE", flush=True)
PY
echo "$(date '+%m-%d %H:%M') voxpopuli download finished" >> logs/round12.log
