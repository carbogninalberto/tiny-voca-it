#!/usr/bin/env bash
# Build the static, browser-only Italian playground (GitHub Pages) into build/pages.
#
#   bash scripts/build_pages.sh
#   PREVIEW_VOICE=runs/voice-two/inference-115000.pt PREVIEW_VOCODER=runs/vocoder-two/vocoder-voicegain.pt PREVIEW_VOICES="IT1 EF3" bash scripts/build_pages.sh
#
# With PREVIEW_VOICE and PREVIEW_VOCODER, a work-in-progress model (voice model and neural
# vocoder in one ONNX graph) is added first in the model list.
#
# - exports the committed Italian checkpoints (models/italian/*.pt) to ONNX + metadata;
# - downloads the pinned eSpeak NG WebAssembly build (npm espeak-ng 1.0.2, GPL-3.0),
#   checks its npm integrity hash and serves it unmodified with its licence;
# - copies playground/web (page, worker, ported front end).
# Requires the Python environment of this repository (torch, onnx); a local espeak-ng is not needed.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/pages
PYTHON=${PYTHON:-.venv/bin/python}
ESPEAK_TARBALL=https://registry.npmjs.org/espeak-ng/-/espeak-ng-1.0.2.tgz
ESPEAK_INTEGRITY=sha512-Xe4YC7d/+O06zYpsqrJ3LpbETdL/IO8JrnAmWcQEMoRFmMLWU+2y2HnpEkOCnqZfb40MBDVyP4ppfusKdWbPcQ==

rm -rf "$OUT"
mkdir -p "$OUT/models" "$OUT/espeak-ng"
cp playground/web/index.html playground/web/worker.js playground/web/frontend.js "$OUT/"
touch "$OUT/.nojekyll"

# eSpeak NG (WebAssembly), verified against the npm registry's integrity hash.
tmp=$(mktemp -d)
curl -fsSL -o "$tmp/espeak.tgz" "$ESPEAK_TARBALL"
actual="sha512-$(openssl dgst -sha512 -binary "$tmp/espeak.tgz" | base64 -w0)"
[ "$actual" = "$ESPEAK_INTEGRITY" ] || { echo "espeak-ng tarball integrity mismatch" >&2; exit 1; }
tar -xzf "$tmp/espeak.tgz" -C "$tmp"
cp "$tmp/package/dist/espeak-ng.js" "$tmp/package/dist/espeak-ng.wasm" "$tmp/package/LICENSE" "$tmp/package/README.md" "$OUT/espeak-ng/"
cat > "$OUT/espeak-ng/SOURCE.md" <<'EOF'
eSpeak NG compiled to WebAssembly, distributed unmodified from the npm package
`espeak-ng` 1.0.2 (https://www.npmjs.com/package/espeak-ng), built from
https://github.com/ianmarmour/espeak-ng.js on eSpeak NG 1.52-dev
(https://github.com/espeak-ng/espeak-ng). Licence: GPL-3.0-or-later (see LICENSE).
Corresponding source: the two repositories above.
EOF
rm -rf "$tmp"

# Models.
export PYTHONPATH=src
# With a preview model, the page offers only it, with the voices in PREVIEW_VOICES (the product
# voices); the released single-voice models are listed only when there is no preview.
LEGACY=()
[ -z "${PREVIEW_VOICE:-}" ] && LEGACY=(tiny-voca-it-IT1 tiny-voca-it-IT2-adapted)
for name in "${LEGACY[@]}"; do
  "$PYTHON" -W ignore -m tiny_voca.export --checkpoint "models/italian/$name.pt" --output "$OUT/models/$name.onnx"
done
PREVIEW_ENTRY=""
if [ -n "${PREVIEW_VOICE:-}" ]; then
  VOICES_JSON=$(printf '"%s", ' ${PREVIEW_VOICES:-IT1 EF3}); VOICES_JSON="[${VOICES_JSON%, }]"
  "$PYTHON" -W ignore -m tiny_voca.export --checkpoint "$PREVIEW_VOICE" --vocoder "$PREVIEW_VOCODER" --output "$OUT/models/tiny-voca-it-preview.onnx"
  STAMP=$(TZ=Europe/Rome date "+%Y-%m-%d %H:%M")
  PREVIEW_ENTRY="  {\"id\": \"preview\", \"file\": \"tiny-voca-it-preview\", \"label\": \"Preview · IT1 + EF3, emotion, neural vocoder · updated $STAMP\", \"voice\": \"IT1\", \"voices\": $VOICES_JSON}"
fi
cat > "$OUT/models/models.json" <<EOF
[
$( [ -n "$PREVIEW_ENTRY" ] && echo "$PREVIEW_ENTRY" || printf '%s\n' \
  '  {"id": "it1", "file": "tiny-voca-it-IT1", "label": "IT1 · Italian base"},' \
  '  {"id": "it2", "file": "tiny-voca-it-IT2-adapted", "label": "IT2 · adapted voice (28 min of data)"}' )
]
EOF
du -sh "$OUT"; find "$OUT" -type f | sort
