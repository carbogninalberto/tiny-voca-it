# Italian training and custom voice plan

Status: planning only. No Italian implementation, dataset download, GPU benchmark,
training run, or voice adaptation has been completed.

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

- [ ] G0: dataset rights/access checked; representative audio and transcripts audited.
- [ ] G1: Italian profile and phonemization work; English artifacts retain their meaning.
- [ ] G2: reproducible manifests have no duplicate leakage and pass acoustic checks.
- [ ] G3: Italian alignments pass vocabulary, duration, and manual boundary audits.
- [ ] G4: a small subset can be learned; representative GAN training fits the GPU.
- [ ] G5: Italian baseline passes held-out intelligibility and listening evaluation.
- [ ] G6: custom-voice adaptation improves resemblance without unacceptable regressions.
- [ ] G7: inference/export metadata and ONNX parity pass; browser remains deferred unless tested.

All gates are pending. Dataset quality, actual GPU throughput, Italian alignment
quality, and the achievable timbre fidelity remain open questions. Fifteen to
thirty minutes of target speech is an exploratory trial; one to three hours is a
planning target, not an established requirement for this architecture.

## Scope of this change

This branch initially adds planning documents only. Future prompts describe work
to implement when requested; they do not start downloads, paid services, training,
deployment, or publication. Keep recordings, checkpoints, annotation manifests,
local reports, and identifying speaker metadata out of Git.
