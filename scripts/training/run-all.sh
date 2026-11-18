#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/import_italian_dataset.py --source data/cml-tts/italian --speaker 1595 --voice IT1 \
    --output data/it-1595-full --keep-splits data/it-1595 > logs/import-it1.log 2>&1
echo IT1-IMPORTED
sed -i '/--speaker 1595/,+1d' logs/import-multi.sh
logs/import-multi.sh > logs/import-multi.log 2>&1 &
logs/step1.sh > logs/step1.log 2>&1
