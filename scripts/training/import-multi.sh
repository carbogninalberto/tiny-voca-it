#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
for spk in 4974 4998 6807 2033 1989 2019 659 10446 1725; do
  $PY scripts/import_italian_dataset.py --source data/cml-tts/italian --speaker $spk --voice CML$spk --output data/it-$spk-full
done
echo IMPORT-DONE
