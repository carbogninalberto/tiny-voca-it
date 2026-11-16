# Italian training and custom voice plan

Status: implemented and trained; see the gates below and [docs/italian.md](../docs/italian.md).

Baseline: `28e0722c6ad040d74a1f6e63d77621272477a71f`.
Working branch: `train-on-it-dataset`.

## Recommended approach

Train the existing small architecture from scratch on one clean Italian voice.
Use the roughly ten-hour Mozilla Italian TTS corpus as the first candidate,
subject to listening, transcript, access, and provenance checks. Use a curated
Italian subset of CML-TTS if that candidate fails. Keep the 24 kHz audio pipeline.

Build an explicit Italian language profile before preparing annotations. Changing
the eSpeak voice alone is insufficient: the current vocabulary, fallback rules,
alignment checks, model buffers, and evaluation assume English.

After the Italian baseline passes intelligibility tests, adapt a separate copy
to a specific speaker using recordings supplied with permission. Compare training
only the speaker embeddings against fine-tuning the acoustic network. A custom
fixed voice is the goal; zero-shot voice cloning is outside this plan.

The target is a desktop RTX 5070 Ti with 16 GB VRAM. Feasibility is a reasonable
engineering expectation, not a measured result. Start with a small frame budget
and measure the full training step, including adversarial losses and validation.

## Documents

- [Implementation plan](IMPLEMENTATION.md): design, data selection, ordered work,
  GPU runbook, voice adaptation, acceptance gates, and rollback.
- [Implementation prompts](PROMPTS.md): coordinator instructions and bounded,
  ordered assignments for future implementation workers.

## Decisions

| Decision | Initial choice | Revisit when |
| --- | --- | --- |
| Base training | Italian from scratch | A comparable Italian checkpoint becomes available |
| Voices | One real speaker | One voice passes the quality gates |
| Dataset | Mozilla Italian TTS female voice | Access, provenance, audio, or alignment audit fails |
| Fallback data | CML-TTS Italian, selected speakers | Required phonetic coverage or usable hours are insufficient |
| Frontend | Italian eSpeak NG profile | Pronunciation audit exposes systematic errors |
| Aligner | Audit the existing multilingual recognizer | Label coverage, timing, or gemination tests fail |
| Precision | Existing full-precision training | Measured memory or speed warrants a separately tested change |
| Deployment | Python first; ONNX second | Browser frontend parity is demonstrated |
| Custom voice | Separate adapted checkpoint | Retaining multiple voices becomes a real requirement |

The corpus choices describe candidates, not downloaded or approved training data.
Do not select a voice based only on its advertised number of hours.

## Completion gates

Status 2026-11-16; evidence and commands in [docs/italian.md](../docs/italian.md).

- [~] G0: Mozilla voice not accessible (sign-in); CML-TTS Italian (CC BY 4.0) used, speaker 1595 chosen by objective audit. Human listening audit pending.
- [x] G1: Italian profile/eSpeak front end, no English fallback, profile digest in checkpoints; English fingerprints unchanged (tests).
- [x] G2: deterministic, order-independent splits by book section with duplicate grouping; test manifest digest locked in the receipt.
- [~] G3: 2.1% rejection, timing probe recorded; voiceless geminates fixed by closure-aware boundaries; sonorant geminate durations still not encoded.
- [x] G4: pilot, GAN ramp and a 100k-step run on the RTX 5070 Ti (4.7 GB / 9.7–10.3 GB peak); resume and adaptation smoke-tested.
- [~] G5: test WER 7.1% vs 9.1% on recordings (ASR gate passes); native listening pending.
- [~] G6: adaptation implemented and tested; demonstrated on a stand-in CML-TTS voice (acoustic beats embedding-only). Real consented speaker pending.
- [ ] G7: export metadata is profile-derived; Italian ONNX parity not yet run; browser deferred.

## Scope of this change

This branch initially adds planning documents only. Future prompts describe work
to implement when requested; they do not start downloads, paid services, training,
deployment, or publication. Keep recordings, checkpoints, annotation manifests,
local reports, and identifying speaker metadata out of Git.
