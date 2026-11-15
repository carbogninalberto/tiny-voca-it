# Implementation prompts

These assignments are pending. They describe future implementation, not permission
to download data, start training, or publish artifacts now. The detailed decisions
and gates are in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Coordinator prompt

```text
Implement the Italian TTS plan in PLAN/IMPLEMENTATION.md, beginning with the first
unfinished dependency. Inspect repository instructions and current changes first;
the plan's baseline is 28e0722c6ad040d74a1f6e63d77621272477a71f, but work may have
advanced. Preserve unrelated work and English artifact compatibility.

You own design, security judgments, review, tests and Git delivery. Have OpenCode
workers on the Go plan with DeepSeek v4.1 Flash write implementation. Supply a
small self-contained assignment, an isolated worktree or exclusive file ownership,
and all relevant contracts. Workers do not inherit this conversation. Do not run
overlapping assignments concurrently. A worker must never commit, push, publish,
access credentials or run unrequested downloads/training. If the model is missing,
ask instead of silently substituting another model.

Review every diff against the surrounding code and acceptance criteria. Run the
targeted tests and appropriate repository checks yourself. Send concrete findings
back for correction. After two unsuccessful rounds on a piece, use a stronger
model only as permitted by the user's implementation policy; report the takeover
and reason. Do not treat worker-reported success as verification.

Adapt the ordered prompts below to the reviewed state. Keep pending gates pending
until there is evidence. Distinguish hypotheses from measured data: no promised
GPU runtime, VRAM limit or voice quality. Full training depends on passing data,
frontend, alignment and representative GPU pilot gates. Custom voice training
depends on an intelligible Italian baseline.

Use git-commit-as-me for verified task-owned changes, including diff review and
secret scans. Keep real audio, checkpoints, local manifests, speaker identities
and private paths out of Git. Confirm the current action request covers acquisition
and GPU runs before performing them; publication/deployment is separate. End each
phase with changed files, checks passed/skipped, measured evidence and open issues.
```

Run a worker from its assigned checkout with the full reviewed assignment replacing
the placeholder below. The placeholder is not an executable implementation task.

```sh
opencode run --model opencode-go/deepseek-v4.1-flash --auto -- '<self-contained assignment>' < /dev/null
```

## 1. Profile foundation

```text
Implement only tiny-voca's language-profile foundation. Read PLAN/IMPLEMENTATION.md
section 2 and inspect src/tiny_voca/{text,config,model,data,align,annotate,refine,
train,infer,export}.py, scripts/export_checkpoint.py and tests/helpers.py.

Add an immutable profile (proposed languages.py) with locale/version, ordered
symbols, special IDs, phone/vowel/voiceless sets and frontend identity. Pass it
explicitly to consumers; imported globals must not decide a loaded model's masks.
Construct Config vocabulary checks, model masks, dataset vowel targets and annotation
checks from the selected profile. Preserve acoustic-feature fingerprint separation.

Preserve the exact English symbol order and legacy frontend fingerprint semantics.
Maintain a format-4 English artifact reader. Missing language metadata means legacy
English only; formats 2/3 remain rejected. If serialization changes require a new
format, implement explicit versioned readers/writers. Record new profile identity
in training/inference checkpoints and export metadata. Reject mismatched profiles
before use; do not bypass checks with allow-frontend-mismatch or strict=False.

Reuse Config.from_dict/to_dict, frontend_identity/fingerprint, export_model,
load_model and journal helpers where appropriate. This phase may provide only the
legacy English profile plus synthetic test profiles. Do not implement Italian
pronunciation or dataset import yet. Keep changes limited to the profile integration.

Acceptance: old English identity/loading preserved; profile/config round-trip;
correct masks; mixed-profile rejection; two synthetic profiles loaded in either
order without cross-contamination. Add meaningful regression tests and run targeted
tests plus PYTHONPATH=src:tests python -m unittest discover -s tests when dependencies
are available. Report missing dependencies; do not install them unasked.

Do not commit, push, publish, access secrets, download data/models or start GPU jobs.
Report changed files, migration decisions, tests actually run, skips and open risks.
```

## 2. Italian frontend

```text
Implement Italian phonemization using the reviewed tiny-voca profile foundation.
Read PLAN/IMPLEMENTATION.md section 3, text.py, the profile module, config.py and
tests/test_frontend.py. Preserve English behavior and the terminal-newline safeguard.

Use Italian eSpeak NG with explicit profile/version identity. Inspect actual IPA
output if eSpeak is installed; otherwise mock the interface and mark real integration
unverified. Do not invent observed output. Do not route Italian through English
dizione or its fallback rules. Preserve Italian r, gn/gl, vowels, affricates, lexical
stress and consonant length; reject/report unknown phones instead of anglicizing them.

Define a provisional gemination representation and its aligner requirements. Keep
TTS-to-aligner mappings explicit and versioned; final inventory is gated on coverage
in assignment 4. Handle accents, elision, abbreviations, decimal commas, currency,
dates, times and punctuation. Any pronunciation overrides must be small, justified,
versioned and tested; spelling alone does not resolve every stress/open-vowel choice.

Reuse chunk/token/stress helpers where correct. Test pala/palla, pena/penna,
nono/nonno, casa/cassa, anno/hanno (homophones), gnocchi, figlio, gli, perché,
città, l'amico, un'altra, 3,14, 12:30, € 25,50 and full sentences containing them.
Test stress/token lengths, special IDs, profile mismatch and unknown handling.

Acceptance: stable Italian identity; no silent English fallback; English regression
tests pass; mocked tests are distinguished from real eSpeak checks; gemination
contract documented. Run targeted frontend tests and the repository suite where
dependencies permit. Limit edits to frontend/profile and tests/docs.

No commits, pushes, publishing, secrets, installation, model/data downloads or
training. Report files, actual observed cases, tests/skips and alignment questions.
```

## 3. Dataset adapter and splits

```text
Implement a local Italian dataset adapter. Read PLAN/IMPLEMENTATION.md section 4,
prepare.py (records/main/extract), journal.py, config.py, tests/test_data.py and
.gitignore. Proposed entry point: scripts/import_italian_dataset.py, with reusable
parsing/splitting helpers. Do not download or inspect private real data.

Primary candidate is Mozilla Italian TTS female voice: WEBM plus TSV and excluded
failed takes. The archive layout has not been audited. Accept explicit column/layout
mappings and fail on ambiguity; do not invent a fixed header. Separate source parsing
from generic validation so a later CML-TTS adapter can reuse it.

Emit prepare-compatible JSONL using id/text/voice_id/audio/split and relative audio
paths. Convert explicitly to mono 24 kHz with a tested decoder/resampler. Record
source checksum, settings, license metadata, seed, exclusions and transformations
in a local receipt. Reuse atomic/journal helpers. Reject unsafe paths/extraction,
bad metadata, missing/corrupt audio, empty text, duplicate IDs, nonfinite/silent
audio and failed takes. Respect 32–3000 frames at hop 256; do not truncate transcripts.

Connect normalized-transcript and decoded-audio duplicates before deterministic
splitting. Input order must not matter. Respect upstream splits while detecting
overlap. Support session/chapter groups where feasible, keeping the training voice
represented; fail clearly on impossible split constraints. Report hours, duration
distribution, overlaps, speakers and coverage. Lock a test-manifest digest.
Resume only matching sources/settings. Preserve raw text and source recordings.

Keep real outputs ignored; extend ignore rules for generated directories/formats
without hiding code/synthetic fixtures. Public reports omit private paths/identities.
Test temporary synthetic media/TSV: reordered input, connected duplicates, leakage,
bad columns, corruption, path traversal, resampling, failed takes, resume and changed
settings. Feed resulting fixture manifests to prepare. Run targeted and repository
tests if dependencies permit; report unavailable codecs as skips.

No commits, pushes, publishing, secrets, dependency installation, downloads or
training. Report changed files, command/schema contract, tests and real-data unknowns.
```

## 4. Italian alignment

```text
Implement tiny-voca's Italian alignment contract without running a full corpus.
Read PLAN/IMPLEMENTATION.md section 5, the reviewed profile, align.py
(PhoneRecognizer/token_spans/assemble), annotate.py, refine.py,
scripts/calibrate_aligner_timing.py and tests/test_alignment.py.

The existing facebook/wav2vec2-lv-60-espeak-cv-ft is a candidate only. Add inventory
coverage preflight, profile-derived phonetic classes, and explicit versioned mappings
if justified. Preserve exact duration/frame sums, positive realized-phone duration
and nonnegative pauses. Test repeated CTC labels, gemination and affricates. Never
collapse Italian length contrasts silently or substitute arbitrary English phones.

Use a cached recognizer only when available and small local fixtures are authorized;
otherwise mock logits and report actual label/timing coverage pending. Do not download
weights. Finalize gemination/inventory only with evidence. If it fails, describe the
blocker and required alternative-aligner adapter instead of claiming success.

Make calibration/QC language-aware. Italian defaults must not assume the English
-19ms onset calibration is validated. Retain journal recovery, fingerprints and
incomplete-run rejection; record mapping/calibration identity. Document the future
100–200-clip pilot and ≥50-clip manual audit. Investigate >10% rejection, never relax
thresholds merely to accept clips. Keep learned refinement off initially; any later
fitting must exclude validation/test data.

Test coverage errors, repeated labels, durations, short audio, unknown phones,
mapping/version mismatch, partial resume and profile invalidation. Verify acoustic
features remain reusable after annotation-only changes. Exercise the NumPy fallback
independently. Run targeted alignment/data tests and the repository suite as possible.

No commits, pushes, publishing, secrets, installation, downloads or real training.
Limit changes to alignment/QC/profile consumers and tests/docs. Report files,
measured checks versus pending data-dependent gates, and next required evidence.
```

## 5. Training measurements and GPU recipe

```text
Prepare a measured RTX 5070 Ti 16 GB training workflow for tiny-voca. Read
PLAN/IMPLEMENTATION.md section 6, train.py (optimize_step/validate/main),
data.py (BucketSampler), losses.py, model.py and training tests. Hardware may be
unavailable; never fabricate benchmark results.

Preserve full-precision training and exact resume. Add minimal logging of peak
allocated/reserved CUDA memory, frames/utterances/audio exposure, step time and
validation overhead without unnecessary per-step synchronization. Make validation
respect a configurable safe frame budget; its separate batching path must be tested.
Use profile-defined speakers and preserve the inference parameter budget.

Document provisional batch-frames 3000/max-batch 8 settings as unmeasured. Provide
synthetic smoke, complete 16–32-clip overfit, representative memory pilot, GAN ramp
and full-run stages. Reference GAN start 8000/ramp 3000; an accelerated smoke GAN
schedule is code-path coverage, not quality evidence. Include long clips, validation
and free-running inference. Do not add AMP/accumulation without measured need.

The runbook must handle max-seconds 3600, last.pt/inference.pt and exact resume;
include required annotation dataset-revision and evaluation tag arguments. Explain
that init-from/new-data retains source step and can retain optimizer/critics, still
requires equal config, and is not speaker migration. Explain total versus extra
steps and smaller-batch exposure. Measure CPU preprocessing/disk/RAM separately.
No runtime extrapolation without a representative pilot.

Add tests for safe validation batching, CPU logging without CUDA, bounded save/stop
and exact resume including sampler/RNG. Run targeted tests and repository suite if
dependencies exist. Run a real GPU pilot only if separately authorized and available;
otherwise leave G4 pending. Limit changes to training, observability, tests/runbook.

No commits, pushes, publishing, secrets, installation, downloads or unrequested
training. Report files, valid commands, tests/skips and hardware-dependent decisions.
```

## 6. Italian evaluation

```text
Implement Italian evaluation and checkpoint-derived voice selection. Read
PLAN/IMPLEMENTATION.md section 7, evaluate.py, infer.py, the profile module and
tests/helpers.py. Preserve English evaluation as an explicit supported mode.

Derive language/speakers from checkpoint metadata; remove hardcoded F2/M2 CLI limits.
Use a deterministic checkpoint-based default and list available voices on bad input.
Italian ASR must use a multilingual model with language it, not small.en/en. Make
model selection configurable and reject English-only models for Italian. Unit tests
must not download or execute real ASR models.

Version Italian scoring normalization and retain raw text. Report synthesis/source
WER and CER using identical scoring, with edit/sample counts, per-speaker/category
results and uncertainty where practical. Create deterministic authored challenges
for stress, gn/gl, gemination, numbers, dates, names, questions and longer clauses.
Keep development and final tests separate; never mutate locked manifests.

Write local structured results and a concise report. Explicitly account for failed
synthesis, skipped ASR and missing audio rather than dropping failures. Support
randomized native-listener ratings for intelligibility/naturalness without committing
real speaker information. Document ≥100 held-out and 50 challenge sentences where
feasible; the plan's thresholds are proposed targets, not achieved scores.

Mock ASR tests for language/model selection, normalization, scoring aggregation,
source baseline, deterministic sampling, configured voices, failure accounting and
manifest immutability. Run targeted and repository tests where dependencies permit.
Limit changes to evaluation/inference and tests/docs.

No commits, pushes, publishing, secrets, installation, downloads or training.
Report files, checks/skips, command contract and requirements for real-checkpoint
and native-listener verification. ASR is not a timbre-similarity metric.
```

## 7. Custom-voice adaptation

```text
Implement a distinct custom-voice adaptation initialization path for tiny-voca.
Read PLAN/IMPLEMENTATION.md section 8, train.py (compatible/load_compatible,
optimizer/checkpoint handling), model.py, config.py, data.py, infer.py and model tests.
Use synthetic fixtures here; real adaptation waits for the Italian baseline gate.

Do not repurpose resume or assume init-from/new-data adds speakers. Require identical
Italian profile/symbol order/DSP and compatible architecture. Default to a target-only
checkpoint; optional retained-speaker support must have an explicit name mapping.
Keep the source unchanged. Strictly copy non-speaker tensors. Migrate BOTH voices
and frame_voices embedding tables by speaker name. Initialize target rows
deterministically from an explicit source row or documented aggregate. Reject
arbitrary missing, extra or shape-mismatched state entries.

Start fresh optimizers, step counter and LR schedule; preserve source step/hash as
provenance only. Default fresh critics to reconstruction warmup and gradual GAN
ramp. Explicitly record any critic-reuse experiment. Emit copied/initialized/frozen/
rejected tensor reports. Resume validates adaptation data, mask, map and schedule.
Document whether initialization accepts full training and/or inference checkpoints.

Provide embedding-only target-row training in both embeddings with all other
model tensors frozen. Prevent retained-row movement from AdamW decay/momentum;
gradient masking is insufficient. Provide separate acoustic fine-tuning with a
declared trainable-module set and initially frozen text encoder. Shared updates can
change old voices; retained-voice mode needs base-data replay and per-voice evaluation.

Test added/reordered/removed voice names, both tables, deterministic initialization,
strict rejection, fresh state, exact resume and byte-identical frozen tensors/rows
after actual optimizer steps. Test target inference with metadata-derived voices.
Run targeted and repository tests. Document clean consented recordings, provisional
15–30min/1–3h targets, held-out sessions/text and blind base/embedding/acoustic
comparison. Optional speaker similarity is diagnostic, not proof.

No commits, pushes, publishing, secrets, installation, downloads or real training.
No personal recordings as fixtures. Report files, supported checkpoint contracts,
tests/skips, migration behavior and unresolved quality requirements.
```

## 8. Export parity and browser audit

```text
Finish Italian export contracts and audit browser readiness. Read
PLAN/IMPLEMENTATION.md section 9, export.py, infer.py, scripts/export_checkpoint.py,
scripts/check_onnx.py, tests/test_export.py, web/public/worker.js,
web/public/index.html and web/fetch_assets.sh.

Derive export example tokens, ordered symbols, masks and voice names from the loaded
profile. Include language, profile/normalizer identity, frontend fingerprint, sample
rate and limits in metadata. Reject mismatched assets/metadata. Preserve English
compatibility. Reuse deterministic external-noise parity checks for waveform,
durations, speed/pitch controls, varied lengths and custom-speaker maps.

Use synthetic English/Italian fixtures. Run actual ONNX Runtime parity if dependencies
exist; otherwise mark it pending. Add negative tests for language/symbol-order/voice
metadata mismatches. Successful serialization alone does not establish parity.

Audit English dizione browser assets and hardcoded F2/M2 UI. Document the deferred
task for a deployable Italian phonemizer, distribution/license review, dynamic voice
UI and Python/browser phone-and-stress parity. Do not implement the browser frontend,
add a service or deploy a site here. Italian browser support remains unavailable
until the separate frontend/parity task passes.

Limit changes to export/inference contracts, tests and readiness documentation.
No commits, pushes, publishing, secrets, dependency installation, model/data downloads
or training. Never commit generated ONNX/weights/audio. Report files, numerical
results actually measured, checks/skips and deferred browser work. Quality still
depends on held-out Italian evaluation, not this export task.
```
