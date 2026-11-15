# Italian training and voice adaptation

All work described here is pending. Baseline:
`28e0722c6ad040d74a1f6e63d77621272477a71f`.
This is an implementation plan, not a record of completed experiments.

## 1. What we are building

First train the existing tiny-voca architecture from scratch on clean Italian
speech from one speaker. Then adapt a separate checkpoint to a specific speaker's
timbre. Preserve the English release and establish Python inference before ONNX
and browser delivery. Zero-shot voice cloning is outside the initial scope.

The [upstream model description](../docs/model.md) reports 4,537,291 inference
parameters, 2,675,312 training-only discriminator parameters, and 17.6 hours of
synthetic English training audio. Its 104,737-step training run reportedly took
7.7 hours on an RTX 4090. These figures do not establish Italian quality or runtime
on a desktop RTX 5070 Ti with 16 GB VRAM.

The architecture predicts duration, pitch, voicing, energy, and source-filter
synthesis parameters. The small synthesizer may capture recognizable timbre while
missing breathiness, texture, or expressiveness. Quality must be measured; adding
more data does not guarantee that this architecture reproduces every voice detail.

## 2. Repository changes and contracts

| Area | Current behavior | Required change |
| --- | --- | --- |
| [text.py](../src/tiny_voca/text.py) | English symbols/fallbacks, `en-us`, English `dizione` | Explicit Italian language profile and phonemizer |
| [config.py](../src/tiny_voca/config.py) | Global 76-symbol vocabulary; F2/M2 defaults | Profile-derived vocabulary and serialized language identity |
| [model.py](../src/tiny_voca/model.py) | Global phone/vowel masks; two speaker embeddings | Profile-derived masks; migrate both `voices` and `frame_voices` |
| [prepare.py](../src/tiny_voca/prepare.py) | JSONL/Parquet, explicit 24 kHz input | Separate dataset import/split adapter |
| [align.py](../src/tiny_voca/align.py) | Multilingual recognizer; English-calibrated timing | Italian label audit and timing calibration |
| [annotate.py](../src/tiny_voca/annotate.py) | Fingerprinted annotation, English phonetic classes | Profile-aware QC and mapping identity |
| [data.py](../src/tiny_voca/data.py) | Global vowel IDs and strict manifest validation | Profile-aware token validation and vowel targets |
| [refine.py](../src/tiny_voca/refine.py) | Global phone classes | Profile-aware refinement, if used |
| [train.py](../src/tiny_voca/train.py) | Frame-budget batches, resume, branched initialization | Measured GPU recipe and explicit adaptation initialization |
| [infer.py](../src/tiny_voca/infer.py) | CLI voices limited to F2/M2 | Load language and voice choices from checkpoint |
| [evaluate.py](../src/tiny_voca/evaluate.py) | English prompts, `small.en`, ASR language `en` | Italian prompts, multilingual ASR and scoring |
| [export.py](../src/tiny_voca/export.py) | Global vocabulary in metadata/examples | Checkpoint-derived metadata and parity tests |
| [browser worker](../web/public/worker.js) | English `dizione` assets | Separate Italian frontend task after Python passes |

Also audit `scripts/export_checkpoint.py`, `calibrate_aligner_timing.py`,
`plot_alignments.py`, and `check_onnx.py`. Synthetic English data generation is not
needed for the real-recording MVP.

### Language profiles and backward compatibility

Introduce an immutable profile, proposed in `src/tiny_voca/languages.py`, containing
locale, version, ordered symbols, special IDs, phone/vowel/voiceless sets, frontend
choice, normalization rules, and explicit canonicalization rules. This filename
and API are proposals, not existing functionality.

Pass profiles explicitly through annotation, datasets, model construction,
inference, and export. Do not rely on changing globals after consumers imported
them. Loading English and Italian models in either order must not affect either
model's token meanings or masks.

Serialize profile identity and digest in new configs, annotations, training
checkpoints, inference checkpoints, and ONNX metadata. Record the phonemizer runtime
version in provenance. Preserve the English symbol order and legacy fingerprints.
Maintain a format-4 English artifact reader; missing language metadata means only
the legacy English contract. If a format bump is needed, implement versioned
readers/writers and migration tests. Formats 2/3 stay rejected.

Preserve acoustic-feature reuse: language/frontend changes invalidate annotations,
not identical wave/mel/pitch features. The acoustic fingerprint should continue to
describe DSP settings. Reject profile mismatches before training/synthesis. Neither
equal vocabulary size nor `--allow-frontend-mismatch` makes language migration safe.

## 3. Italian phonemization

Use eSpeak NG's Italian voice as the first frontend, with a pinned version. Inspect
its real IPA output before freezing the inventory. The English fallback table must
not canonicalize Italian: it maps /r/ to English /ɹ/, /e/ to /eɪ/, /o/ to /oʊ/,
and /ɲ/ and /ʎ/ to approximations.

Preserve Italian vowels including open/closed e/o, /r/, /ɲ/, /ʎ/, affricates,
stress, and consonant length. Choose one documented representation for gemination
and prove that it aligns. Long-phone tokens are suitable only if recognizer labels
support them; otherwise specify a tested mapping and duration aggregation. Repeated
identical CTC labels need blank separation. Do not silently collapse long consonants.
If mapping remains unreliable, change aligner before full annotation.

Handle accented letters, apostrophes/elision, abbreviations, decimal commas, dates,
currency, times, and punctuation. Preserve the existing terminal-newline safeguard.
Treat ambiguous stress/open vowels as a pronunciation issue, not a problem solved
by spelling alone. Add a versioned override mechanism only when justified.

Test words and full sentences containing `pala/palla`, `pena/penna`, `nono/nonno`,
`casa/cassa`, `anno/hanno`, `gnocchi`, `figlio`, `gli`, `perché`, `città`,
`l'amico`, `un'altra`, `3,14`, `12:30`, and `€ 25,50`. `anno/hanno` is an intentional
homophone test. Include unknown-phone rejection, stress/token length consistency,
and a mixed English/Italian model-loading regression test.

Gate G1: no silent English fallback for Italian; stable profile identity; tested
pronunciation contract; old English artifacts retain their meanings. Finalize the
Italian inventory only after the alignment label audit.

## 4. Dataset choice and acquisition gate

### First candidate

[Mozilla Italian TTS female voice](https://mozilladatacollective.com/datasets/cmoiuyem401j5mr07s0jx8rqr)
is listed as roughly ten hours of single-speaker Italian, WEBM audio with TSV
transcripts, under Apache-2.0. The description notes a slight Abruzzo accent and
failed takes in a separate folder. The archive has not been downloaded or heard
for this plan; verify access, provenance and terms at acquisition.

Audit at least 100 clips distributed across sessions/source sections where possible.
Listen for noise, compression artifacts, clipping, inconsistent distance and reading
errors. Compare audio with transcripts. Explicitly exclude failed takes. Record
usable hours after filtering rather than repeating the advertised duration.

### Fallback

[CML-TTS](https://github.com/freds0/CML-TTS-Dataset) provides Italian audiobook
speech for TTS at 24 kHz under CC BY 4.0, with downloads at
[OpenSLR](https://www.openslr.org/146/). Select one consistent speaker initially.
Check actual per-speaker usable hours and preserve attribution. Do not assume the
whole Italian collection belongs to one speaker or mix inconsistent recordings
just to increase hours.

Gate G0: source access and rights verified, sample audit documented, supported
archive/transcript schema inspected, and a candidate selected with reasons.
If the first candidate fails, repeat the audit on the fallback.

### Import contract

Implement a separate adapter, proposed as `scripts/import_italian_dataset.py`, with
reusable parsing/splitting helpers. Keep source-specific parsing separate from
generic validation. Discover the real TSV/archive layout; do not invent headers.
Allow explicit column/layout mapping and reject ambiguous inputs.

1. Record URL, source revision or archive checksum, license, selection rules, seed,
   decoder/resampler versions, and transformations in a local immutable receipt.
2. Decode WEBM using a tested tool such as FFmpeg, explicitly convert to mono, and
   resample once to 24 kHz. Avoid aggressive denoising/pitch/loudness processing that
   can change voice identity. Audit transformed audio.
3. Emit stable IDs and existing default columns: `id`, `text`, `voice_id`, `audio`,
   `split`. JSONL audio should be a path relative to the manifest; `prepare.py`
   also accepts an audio object with a path or Parquet binary bytes.
4. Reuse `prepare.records()` and journal/atomic helpers where appropriate. Populate
   supported provenance keys; retain additional provenance in the local receipt.
5. Reject unsafe archive extraction/paths, duplicate IDs, empty text, missing or
   corrupt files, non-finite audio, silence, and unusable metadata with reasons.
6. Respect 32–3,000 frames at hop 256, approximately 0.34–32 seconds at 24 kHz.
   Prefer shorter sentence clips. Split long audio only at verified transcript
   boundaries; never truncate audio while retaining the full transcript.
7. Resume only when inputs and settings match. Preserve source data unchanged.

### Deterministic splits

Start near 90/5/5 by usable duration, adjusting for groups. Before assigning splits,
connect records sharing normalized transcripts, identical decoded audio hashes, or
known duplicate takes. Place the entire connected group in one split. Detect near
duplicates where practical. Preserve original text separately from deduplication
text. Input order must not change the split.

Group by chapter/session when enough independent groups exist. Do not hold out the
only training speaker: a fixed-voice model needs that speaker represented in both
training and evaluation. Preserve appropriate upstream splits and audit cross-split
duplicates. Report hours, clips, duration quantiles, speaker coverage and phone
coverage; rare phones/geminates must not all land outside training.

Lock the test manifest checksum before model selection. Use validation for tuning;
do not change the test set after hearing a weak checkpoint.

Gate G2: reproducible preparation, zero duplicate leakage, accepted acoustic
features, documented coverage, and locked evaluation manifests.

### Storage and handling

Keep real raw/processed audio, features, annotations, weights, target-speaker data,
local reports and identifying metadata out of Git. Extend ignore rules for generated
directories and formats such as WEBM/TSV where necessary; existing rules are not
complete. Commit synthetic fixtures, code, schemas, and sanitized aggregate reports
only. Keep permissions/provenance records private when they identify a speaker.

Measure storage and RAM on a pilot. Ten hours of 24 kHz mono float32 waveform alone
is about 3.46 GB, before source archives, copies, features and checkpoints. CPU
`librosa.pyin` can dominate preparation. Training has `--no-preload` if preloading
features exceeds available RAM. Neither a disk budget nor preparation time has
been measured for the proposed corpus.

## 5. Alignment gate

The existing `facebook/wav2vec2-lv-60-espeak-cv-ft` is a candidate multilingual
recognizer, not a validated Italian aligner. It loads from the local cache; obtain
a pinned revision through an authorized acquisition step before offline annotation.
Annotation runs separately from training, so their GPU memory need not coexist.

Before full annotation:

1. Audit every Italian phone against the recognizer's labels and test any mapping,
   particularly gemination and affricates. Freeze the inventory/mapping together.
2. Annotate 100–200 diverse development clips. Inspect failures, duration quantiles,
   one-frame phones, pauses, and voiced-vowel coverage by phone and speaker.
3. Manually inspect at least 50 clips using plots and listening. Recalibrate the
   English -19 ms onset offset against Italian acoustic boundaries.
4. Require no unknown phones in accepted samples, exact duration/frame sums,
   positive realized-phone durations and nonnegative pauses.
5. Investigate pilot rejection above 10% as a provisional stop condition. Review
   systematic per-phone errors even at lower rates; do not loosen QC just to pass.

Existing QC thresholds are initial candidates, not validated Italian settings.
Record changes and evidence. If persistent Italian errors remain, evaluate a
language-capable alternative with an explicit phoneme-duration adapter. Word-level
timestamps are not a drop-in substitute.

Preserve annotation identity, journals, recovery and completion flags. Limited
annotation runs are incomplete and cannot feed the strict training loader; make a
complete small dataset for overfit tests. Keep learned refinement off initially.
If used later, fit it on training data only, then apply it without fitting held-out
audio. Include profile/mapping/calibration identity in annotation provenance.

Gate G3: coverage, timing, gemination and rejection audits pass; failed alignments
are explained; the final manifests are complete and reproducible.

## 6. Training on the RTX 5070 Ti

### Environment

Use Linux or WSL2 initially; `prepare.py` imports `fcntl`. Confirm the actual GPU
and VRAM. Pin a compatible NVIDIA driver, PyTorch CUDA build, matching torchaudio,
Python and preprocessing dependencies on the training host. Blackwell support
arrived with PyTorch 2.7/CUDA 12.8, but that is historical context, not advice to
use an old release or assume every wheel supports this GPU.
[PyTorch announcement](https://pytorch.org/blog/pytorch-2-7/).

Test CUDA allocation, matrix operations, FFT/complex DSP, forward/backward, and
alignment API availability. Test the NumPy alignment fallback independently rather
than assuming a removed torchaudio API still exists. Record actual versions locally.

The existing test command, after dependencies are installed, is:

```sh
PYTHONPATH=src:tests python -m unittest discover -s tests
```

This is a command to execute later, not evidence of a passing GPU environment.

### Experiment sequence

| Stage | Scope | Required evidence |
| --- | --- | --- |
| Dependency smoke | Synthetic fixture | Finite forward/backward and checkpoint round-trip |
| Overfit | 16–32 complete clean clips | Falling loss and recognizable training phrases |
| Memory pilot | Short and long representative batches | Measured peak VRAM/RAM and step time |
| GAN pilot | Before activation, through ramp, full weight | Stable finite losses/audio and memory headroom |
| Development run | Curated Italian corpus | Improving validation/listening at milestones |
| Final evaluation | Selected checkpoint on locked test | WER/CER, listening and explicit limitations |

Provisional initial options: `--batch-frames 3000 --max-batch 8`. These are not
verified memory limits. Padded frame budget is not GPU bytes. Validation currently
has a separate batching path; make its frame budget safe too. Measure long clips,
validation and free-running inference. Lower batch/frame limits on OOM first.

The reference GAN starts at step 8,000 and ramps over 3,000 steps. Profile through
the full ramp; a pre-GAN memory check is insufficient. An early GAN boundary in a
throwaway smoke run exercises code but does not establish training quality.

Keep current full precision initially. There are no existing AMP or gradient-
accumulation CLI options. Add either only if measurements justify it, with separate
tests; FFT/complex synthesis and spectral losses need special precision care.

Log clips/audio-seconds processed, actual batch distribution, losses, validation,
learning rate, peak allocated/reserved VRAM and throughput. Smaller batches process
less data per step, so 104k upstream steps are only a reference. Choose exposure
and stopping criteria using validation rather than blindly copying that count.

### Checkpoints and stopping

The trainer defaults to `--max-seconds 3600`, writes `last.pt` and `inference.pt`,
and supports bounded resumable invocations. Preserve milestone and best-validation
checkpoints outside Git. Keep a receipt linking source data, code, environment,
profile, recipe and chosen checkpoint; document the selection reason.

- `--resume` restores the same run's optimizer, critics, RNG, sampler and step.
- `--init-from` branches while retaining source step and potentially optimizer
  moments and trained critics. `--steps` is a total endpoint, not extra steps.
- `--new-data` relaxes annotation/frontend matching on that branch path but still
  requires equal configuration. It does not safely add speakers or languages.

Stop on NaNs, persistent OOM, unstable audio or sustained validation regression.
Preserve diagnostics locally; fix the cause and use a new identity for changed
data/config/recipe. Never overwrite the known-good checkpoint or reinterpret IDs.

After a representative pilot, estimate runtime from pre/post-GAN step costs,
validation overhead and remaining exposure. Report CPU preparation separately.
No 5070 Ti wall-time estimate is established yet.

Future executable runbooks must include valid config/data paths and all required
arguments, including `annotate --dataset-revision` and `evaluate --tag`, which the
short README example omits. Do not present proposed Italian flags/files as available
before implementation and smoke verification.

Gate G4: complete subset can be learned; representative training, GAN, validation
and inference fit the GPU with measured headroom and checkpoint recovery works.

## 7. Italian quality evaluation

Use multilingual ASR, for example Whisper `small`, with language `it`, not
`small.en`/`en`. Version an Italian scoring normalizer and preserve raw transcripts.
Report WER and CER on both source and synthesized audio using the same scorer;
the source score exposes ASR's error floor. ASR does not measure naturalness or
voice identity.

Use at least 100 held-out sentences where feasible, plus 50 authored challenge
sentences covering stress, gn/gl, geminates, numbers, dates, names, questions and
longer clauses. Report per-speaker/category counts and errors. Maintain separate
development challenges and locked final tests. Select checkpoints on validation.

Proposed gates to agree before the full run, not established results:

- No non-finite output, crashes, persistent silence or systematic truncation.
- Test synthesis WER at most 10% and at most 5 percentage points above source-audio
  WER. Include raw edit counts and uncertainty; investigate poor source ASR first.
- Native listeners find at least 90% of a stratified sample understandable without
  transcripts. Flag systematic phone errors even if aggregate WER passes.
- Score naturalness separately: 1 unintelligible, 2 major defects, 3 understandable
  with artifacts, 4 mostly natural, 5 natural. Mean 3 is an initial experiment target.

Use at least two native listeners where possible, otherwise label findings as a
single-listener pilot. Preserve error examples locally. Do not present a few good
sentences as a benchmark or conflate incompatible scoring normalizers.

Gate G5: held-out and listening results support an intelligible Italian baseline,
with failure cases and limitations recorded.

## 8. Custom voice timbre adaptation

This phase starts after G5. Use recordings and transcripts from a speaker who has
granted permission for the intended use; keep permissions and identifying records
private. Pin the base checkpoint hash and preserve it unchanged. Fifteen to thirty
minutes is exploratory; one to three hours is a planning target, not a proven
tiny-voca requirement. Hold out different sentences and preferably a different
session. Include broad phonetic coverage rather than repeated short phrases.

### New-run initialization contract

Implement a dedicated adaptation mode; its CLI does not exist yet. Do not overload
exact resume or assume the existing init-from options provide this behavior.

1. Require identical Italian profile/symbol order/DSP and compatible architecture.
   Fail with a precise mismatch report instead of permissive state loading.
2. Support a standalone target-voice checkpoint by default. An expanded voice list
   is optional and requires old-voice retention work.
3. Copy compatible non-speaker tensors strictly. Migrate both `voices.weight` and
   `frame_voices.weight`, mapping retained rows by speaker name, not index alone.
4. Initialize target rows deterministically from a selected source row or documented
   aggregate. Record the decision; an untrained row does not already encode timbre.
5. Start fresh optimizers, adaptation step and LR schedule. Keep source step only
   as provenance. Default to fresh critics, reconstruction warmup and gradual GAN
   ramp. Reuse trained critics only as an explicit comparison experiment.
6. Emit a copied/initialized/frozen/rejected tensor report. Exact adaptation resume
   validates data, profile, speaker map, schedule and trainable-parameter mask.

### Controlled comparisons

- Base checkpoint on the same held-out text.
- Embedding-only adaptation of the target rows in both embeddings, freezing all
  other model parameters. Prevent AdamW decay/momentum from changing retained rows;
  gradient masking alone is insufficient. Test actual optimizer steps.
- Acoustic fine-tuning of target embeddings and frame/acoustic/synthesis modules,
  initially freezing the text encoder. Try conservative LR candidates such as 1e-5
  and 3e-5 on validation; they are proposals, not established optimal values.

If retaining source voices, shared-weight updates can alter them even with frozen
embedding rows. Add representative base-data replay and evaluate every voice.
Target-only adaptation does not need to preserve source-voice behavior.

Compare timbre using randomized blind listening on unseen sentences, with held-out
real recordings as references and consistent playback loudness. Score resemblance,
naturalness and intelligibility separately. Speaker-encoder similarity can be an
optional diagnostic, never proof of identity. Prefer adaptation only if resemblance
improves without unacceptable intelligibility loss; an initial tolerance is at most
2 percentage points WER regression versus the base on matched evaluation text.
Report disagreements and uncertainty for small samples.

Gate G6: strict migration/resume tests pass and held-out comparisons demonstrate
better resemblance with acceptable intelligibility. Do not claim success from a
memorized training phrase or a changed pitch alone.

## 9. Inference, ONNX and browser

Python inference derives language and voice names from the checkpoint, reports
available voices on invalid input, and uses a deterministic checkpoint-based default.
Keep English tests and add Italian/custom-speaker fixtures.

ONNX metadata must include profile/normalizer identity, ordered symbols, speakers,
sample rate, limits and frontend fingerprint. Derive example inputs from that profile.
Reuse `scripts/check_onnx.py` for deterministic external-noise waveform/duration
parity, speed and pitch controls, and varied lengths. Test both languages; exporting
successfully is not parity evidence.

The browser currently loads English `dizione` assets and hardcodes F2/M2 in its UI.
An Italian ONNX model alone is insufficient. Defer browser implementation until
Python quality passes. Then choose a deployable Italian frontend, review distribution
obligations and require Python/browser phone-and-stress parity on shared fixtures.
eSpeak NG is GPL-licensed; document its role rather than assuming the English
frontend's permissive packaging carries over. Deployment is a separate action.

Gate G7: correct metadata and Python/ONNX parity. Browser support stays explicitly
unavailable until its separate frontend, voice UI and parity work passes.

## 10. Delivery, tests and rollback

Use the ordered assignments in [PROMPTS.md](PROMPTS.md): profile foundation,
Italian frontend, dataset adapter, alignment, training/runbook, evaluation,
custom voice, export/browser audit. Files overlap, so do not run these workers
concurrently on one checkout. Evaluation must be implemented before full training
selection; adaptation starts only after G5. Browser implementation is deferred.

Each phase must be reviewable with synthetic fixtures, without downloading a corpus.
Add tests for changed contracts and opt-in integrations for cached models/GPU/data.
Report skipped checks explicitly. Important coverage includes old artifacts, profile
isolation, duplicate leakage, import recovery, duration conservation, both speaker
tables, frozen tensors after optimizer steps, exact resume and ONNX parity.

Rollback selects the last passing code/checkpoint combination together with its
config, environment, profile, manifests and digests. Rebuild annotations when the
frontend/aligner changes; reuse acoustic features only if their DSP identity matches.
Never overwrite raw data or reuse an output directory with a mismatched identity.

### Required evidence

- [ ] Dataset receipt, usable hours, exclusions, attribution and split audit.
- [ ] Profile/normalizer/aligner versions, phone coverage and timing audit.
- [ ] CPU preparation, storage/RAM, CUDA versions and GAN peak VRAM measurements.
- [ ] Training curve, exposure, resume check and checkpoint selection rationale.
- [ ] Test WER/CER, source ASR baseline, native listening and error categories.
- [ ] Voice-adaptation migration and held-out comparison, if attempted.
- [ ] English compatibility, Italian inference and ONNX parity checks.
- [ ] Sanitized documentation stating limitations and unverified claims.

All items are pending. Publishing datasets, recordings, weights or a web demo is
outside this planning task.
