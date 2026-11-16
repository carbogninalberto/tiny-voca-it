# Italian tiny-voca: runbook and status

This records what the Italian pipeline does, the commands that produced the first
Italian model, and what was measured. Unmeasured items are marked as such.
The plan and gates are in [PLAN/IMPLEMENTATION.md](../PLAN/IMPLEMENTATION.md).

## Environment (measured 2026-11-15)

- Host: WSL2, NVIDIA GeForce RTX 5070 Ti 16 GB (driver 595.97, CUDA 13.2), 12 CPU threads, 15 GB RAM.
- Python 3.12.3, PyTorch 2.11.0+cu128, torchaudio 2.11.0+cu128 (`forced_align` present).
  Checked: CUDA matmul, STFT/complex FFT, forward/backward of the full training step with GAN.
- eSpeak NG 1.51 (Ubuntu `espeak-ng 1.51+dfsg-12build1`), unpacked without root into
  `~/.local/espeak` and wrapped by `~/.local/bin/espeak-ng` (sets `LD_LIBRARY_PATH`,
  `ESPEAK_DATA_PATH`). eSpeak NG is GPL-3.0; it runs as a separate program and is not
  distributed with the model.

```sh
python3 -m venv .venv
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -e ".[prepare,annotate,evaluate]" onnx onnxruntime
export PATH=~/.local/bin:$PATH      # espeak-ng wrapper
```

## Language profile

`src/tiny_voca/languages.py` defines immutable profiles. Every consumer takes the
profile from `Config.language` (`Config.profile`): model masks, dataset vowel
targets, alignment classes and labels, annotation QC, refinement classes,
inference, export metadata and evaluation.

- English (`en`) is the legacy format-4 contract: same 76 symbols, fallback table and
  front-end fingerprint. `language` is omitted from English configs, so existing
  configs, checkpoints and fingerprints are unchanged.
- Italian (`it`, profile version 1): 66 symbols = 5 specials + 12 punctuation + 42
  consonants (24 short, 18 long) + 7 vowels (`a e ɛ i o ɔ u`).
  - Geminates are one long token (`palla` = `p a lː a`). eSpeak 1.51 writes them as
    `l_l`, `ss` or `kː`; all become the long token. Identical consonants across a word
    boundary are not merged.
  - eSpeak's `eɪ aɪ aʊ …` for Italian `dei`, `mai`, `causa` are two segments.
    `ɪ ʊ` → `i u`; vowel length is dropped (not phonemic).
  - No English fallback: other phones become `<unk>` and annotation rejects the clip.
  - Italian checkpoints store `language` metadata with the profile digest; a digest
    mismatch is an error. Missing metadata means legacy English only.
  - Abbreviations such as `Sig.`, `dott.`, `ecc.` are expanded before eSpeak.

Corpus check: of 5,563 transcripts, 5 still contain an unknown phone (`h` in English
names such as "Harry"); they are rejected at annotation. `ʃː` and `ʎː` never occurred
(eSpeak writes them short) and are not in the inventory.

## Data (gate G0/G2)

The first candidate, the Mozilla Italian TTS voice on Mozilla Data Collective,
requires a signed-in account to download, so it was not used. The fallback, CML-TTS
Italian (CC BY 4.0, LibriVox audiobooks, 24 kHz), was taken from the Hugging Face
mirror `ylacombe/cml-tts` at revision `48aedd7a8ea08efc75cae3e81fbefa9720df0812`
(65 Parquet shards, 22 GB, 145 h, many speakers).

Speaker choice: an objective audit of 40 clips each from the four largest speakers
(Whisper `small`, language `it`, on 12 clips each; SNR as 95th−10th percentile frame
level; clipping; pitch):

| speaker | hours | WER | SNR dB | median F0 |
|---|---|---|---|---|
| **1595** | 24.4 | **0.108** | 38 | 152 Hz |
| 4974 | 17.8 | 0.185 | 56 | 168 Hz |
| 4998 | 12.4 | 0.164 | 49 | 122 Hz |
| 6807 | 10.3 | 0.201 | 43 | 178 Hz |

Speaker 1595 was chosen (most intelligible, most data, no clipping). No human
listening audit has been done yet; it is still required by G0.

```sh
python scripts/import_italian_dataset.py --source data/cml-tts/italian --speaker 1595 \
    --voice IT1 --output data/it-1595 --max-hours 14
```

Filters: transcript/ASR similarity ≥ 0.93 (drops misreadings), 1–30 s, 14 h cap
(seeded). Groups: book section (100 consecutive clips) joined with identical
normalised transcripts and identical audio; 100 groups. Result:

| split | clips | hours |
|---|---|---|
| train | 4,982 | 11.52 |
| validation | 266 | 0.65 |
| test | 315 | 0.69 |

The receipt (`data/it-1595/receipt.json`, local) holds shard checksums, settings,
rejections and the locked test-manifest digest.

## Features

```sh
for split in validation test train; do
  python -m tiny_voca.prepare --inputs data/it-1595/$split.jsonl --split $split \
      --output data/it-features --config configs/italian-it1.json --workers 12
done
```

`--workers` (new) runs decoding and `librosa.pyin` in processes; journals and resume
are unchanged. Measured: validation (0.65 h) 4 min 10 s with 11 workers, 650 MB RSS.

## Alignment (gate G3)

The recogniser `facebook/wav2vec2-lv-60-espeak-cv-ft` (revision `ae45363b`) has a label
for every Italian phone except `dz`, which is aligned as `dzː`. Long consonants are
aligned with their short label. A 150-clip pilot compared short labels against the
recogniser's own long labels (`kː`, `tː`, …):

| mapping | mean token prob. | weak phones | geminate prob. |
|---|---|---|---|
| short labels (chosen) | 0.753 | 7.2% | 0.873 |
| long labels where available | 0.729 | 10.0% | 0.313 |

Limitation: in that pilot, aligned long consonants were only slightly longer than
short consonants on average (5.4 vs 5.0 frames). The onset offset is the English −19
ms value; Italian recalibration and a 50-clip manual boundary audit are still pending.

### Full annotation (measured)

```sh
python -m tiny_voca.annotate --features data/it-features --output data/it-annotations \
    --dataset-revision cml-tts-48aedd7a-speaker1595 --config configs/italian-it1.json
```

4.5 minutes on the RTX 5070 Ti. Existing QC thresholds were kept unchanged.

| split | accepted | rejected | hours | reasons |
|---|---|---|---|---|
| train | 4,878 | 104 (2.1%) | 11.38 | overlong phone 38, weak alignment 38, low confidence 31, vowels unvoiced 12, unknown phone 5 |
| validation | 261 | 5 | 0.64 | |
| test | 314 | 1 | 0.69 | |

Median QC (train): token probability 0.77, weak phones 6.9%, no one-frame phones,
13.3 phones/s, vowel frames voiced 88%. Label-free timing probe: vowel→voiceless
boundaries sit +0.64 frames (≈7 ms) after the voicing flip on average (77% within 2
frames); fricative→vowel +0.15 frames (95% within 2 frames).

**Gemination timing: G3 is not met.** Mean aligned frames, short vs long:
`l` 5.4 / `lː` 3.8, `n` 5.6 / `nː` 4.0, `t` 4.3 / `tː` 4.0, `k` 4.2 / `kː` 4.5,
`p` 4.0 / `pː` 4.2, `r` 5.4 / `rː` 5.8, `s` 8.6 / `sː` 11.0. The recogniser marks
phone onsets (a stop at its release), so the closure of a geminate is assigned to
the preceding vowel. Durations still sum exactly and the long token is distinct, so
the model can learn the contrast from context, but the duration targets do not
encode it. Next step: closure-aware boundary refinement (move a consonant's start back
to the energy drop inside the preceding vowel) or the learned refinement pass, then
re-annotate and compare the minimal pairs.

## Training

Recipe: upstream defaults with `--batch-frames 9000 --max-batch 32 --gan-start 8000
--gan-ratio 0.7 --teacher-final 0`, full precision. Measured on the RTX 5070 Ti with
the real Italian data:

| phase | steps/s | audio s/s | peak reserved VRAM |
|---|---|---|---|
| before the GAN (steps 0–8,000) | 9.7–10.2 | ~820 | 4.7 GB |
| GAN (from step 8,000, through the ramp) | 4.3–4.5 | – | 9.7 GB |

Validation passes take about 0.5 s. A 100,000-step run is therefore about 6 hours.
Preloaded features use about 6 GB of RAM.

`runs/it-base` (first annotations, stopped at step 8,700) was the pilot: losses
finite, GAN stable (discriminator ≈ 0.46, adversarial ≈ 0.30, gradient-balance scale
≈ 4.4). The main run `runs/it-v2` uses the closure-corrected annotations:

```sh
python -m tiny_voca.train --train data/it-annotations-v2/train.jsonl \
    --validation data/it-annotations-v2/validation.jsonl --output runs/it-v2 \
    --config configs/italian-it1.json --steps 100000 --batch-frames 9000 --max-batch 32 \
    --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 --max-seconds 86400 \
    --save-every 1000 --keep-every 10000
# continue after an interruption (same arguments) with --resume runs/it-v2/last.pt
```

## Evaluation

```sh
python -m tiny_voca.evaluate --checkpoint runs/it-v2/inference.pt --annotations data/it-annotations-v2 \
    --output eval/it-v2 --tag it-v2 --splits validation test --per-voice 100 --device cuda
```

Italian evaluation uses faster-whisper `small` (multilingual, language `it`; `*.en`
models are refused), the versioned normaliser `it-norm-1` and 25 authored challenge
sentences (geminates, homophones, gn/gl, stress, affricates, numbers, dates, names,
elision, questions, long clauses).

Pilot result, `runs/it-base` step 8,000 (before the GAN), 10 clips per split:

| set | synthesis WER (pooled) | recording WER | synthesis CER |
|---|---|---|---|
| validation | 0.115 | 0.162 | 0.020 |
| test | 0.078 | 0.091 | 0.015 |
| 25 challenge sentences | 0.020 | – | 0.013 |

The recordings score worse than the synthesis because the transcripts are archaic
literary text. Most remaining "errors" are number/currency formatting (`cinque` → `5`,
`€` → `euro`). Two are real geminate errors: `pala` was heard as `palla`, and
`arrivammo` as `arrivamo`. Small samples: these are pilot numbers, not a benchmark.
No native-listener evaluation has been done.

## Playground

```sh
python scripts/playground.py --checkpoint it-v2=runs/it-v2/inference.pt --device cpu
# open http://127.0.0.1:8765
```

`playground/index.html` is served by `scripts/playground.py`, which phonemizes with
eSpeak NG and synthesises in Python (the browser cannot run the Italian front end).
Checkpoints reload when the file changes, so a running training can be followed live.
Controls: model, voice, speed, pitch (semitones), intonation range, noise seed.
Long text is read sentence by sentence.

## Results (2026-11-16)

`runs/it-v2` trained 100,000 steps in about 6 h 15 min. Checkpoints were selected on
VALIDATION only (100 clips each, faster-whisper `small`, `it-norm-1`); the recordings
score 11.9% WER on the same clips:

| step | WER | CER | mel DTW | length ratio |
|---|---|---|---|---|
| 60k | 9.1% | 1.73% | 0.748 | 0.94 |
| 70k | 9.3% | 1.84% | 0.748 | 0.98 |
| 80k | 9.1% | 1.70% | 0.756 | 1.01 |
| **90k (selected)** | **8.5%** | **1.64%** | 0.772 | 0.98 |
| 100k | 8.9% | 1.69% | 0.757 | 0.99 |

Differences are within noise; 90k has the best combined rank. Locked test set (all
314 clips, evaluated once, for the selected checkpoint only):

| | synthesis | recordings |
|---|---|---|
| WER (pooled) | **7.1%** | 9.1% |
| CER | **1.6%** | 2.8% |

The plan's ASR gate (≤ 10% WER and ≤ 5 points above the recordings) passes. The front end
reproduced the annotated tokens for every test clip. 25 challenge sentences: 4.5% WER.
Real errors are geminates (`pala` → `palla`, `pena` → `appena`, `arrivammo` → `arrivamo`,
all singleton/long sonorants or vowel+sonorant) and proper names (`Garibaldi` →
`Garibaldia`). The others are number/currency formatting. Mean pitch: synthesis 149 Hz,
IT1 recordings 148 Hz.

Still open: native-listener intelligibility and naturalness ratings (G5 listening
part), sonorant gemination (`lː nː mː rː` durations; see alignment), ONNX parity for
the Italian model (G7) and a browser Italian front end (deferred by the plan).

## Custom voice adaptation (demonstration)

No consented target-speaker recordings were available, so CML-TTS speaker 4998 (a
different, lower voice) stood in: 0.6 h imported, 170 train clips (28 min) accepted,
11 validation, 14 test.

```sh
python scripts/import_italian_dataset.py --source data/cml-tts/italian --speaker 4998 --voice IT2 \
    --output data/it-4998 --max-hours 0.6 --section-size 15
# prepare + annotate with configs/italian-it2.json (speakers: IT2), as above
python -m tiny_voca.train --adapt-from runs/it-v2/inference.pt --init-voice IT1 --adapt-mode acoustic \
    --config configs/italian-it2.json --train data/it2-annotations/train.jsonl \
    --validation data/it2-annotations/validation.jsonl --output runs/it2-adapt-acoustic \
    --steps 6000 --learning-rate 3e-5 --warmup 200 --gan-start 1000 --gan-ramp 1000 \
    --teacher-hold 300 --teacher-ramp 700 --teacher-final 0 --validation-size 11
```

`--adapt-from` copies every non-speaker tensor exactly, migrates both speaker tables by
name (a new voice starts from `--init-voice`), and starts the optimizer, critics and
step counter fresh. Frozen tensors are left out of the optimizer, so weight decay
cannot move them. Source hash, step, mode and tensor report are in `recipe.json` and
the inference checkpoint's `provenance`. Measured: acoustic 6,000 steps ≈ 23 min,
embedding-only 3,000 steps ≈ 10 min.

| (IT2 held-out clips) | acoustic (2.95 M trained) | embedding only (1.5 k trained) | IT2 recordings |
|---|---|---|---|
| mean pitch | **100 Hz** | 133 Hz | 107 Hz |
| mel DTW to the recording | **0.95–0.99** | 1.04–1.05 | – |
| test WER (14 clips) | 13.0% | 13.9% | 15.2% |
| challenge WER (25) | 2.9% | 4.1% | – |

The base voice IT1 sits at about 150 Hz, so acoustic adaptation moves most of the way
to the new speaker at the same intelligibility. Embedding-only moves about a third of
the way. Validation mel of the acoustic run was lowest at step 500 and drifted up
slightly (0.81 → 0.83), a sign of mild overfitting on 28 minutes (11 validation clips
are a noisy measure). For a real speaker: record 1–3 h if possible, keep a separate
validation session, and select the adaptation checkpoint on validation. Resemblance
must be judged by blind listening against held-out real recordings; these numbers are
diagnostics, not proof of identity.

## Final playground

```sh
logs/run-playground.sh     # or: python scripts/playground.py --host 0.0.0.0 --checkpoint name=path ...
```

Serves `IT1 base (step 90k)`, `IT2 adapted - acoustic`, `IT2 adapted - embedding only`
and `IT1 pilot (step 8k)` from `runs/release/` (local, not in Git) on port 8765. For
access from the local network under WSL2 (NAT), Windows needs a port proxy and a
firewall rule (elevated PowerShell):

```powershell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8765 connectaddress=<WSL IP> connectport=8765
New-NetFirewallRule -DisplayName "tiny-voca playground" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow -Profile Private,Domain
```
