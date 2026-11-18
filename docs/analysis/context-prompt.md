# Context prompt: tiny-voca-it, where we are (18 November 2026)

Paste this into a new session to continue the work.

---

You are continuing work on **tiny-voca-it**, a small Italian text-to-speech model.

- **Repository:** https://github.com/carbogninalberto/tiny-voca-it, branch `neural-vocoder-emotion`.
- **Local path:** `/home/alberto/repos/tiny-voca-it` (WSL2, RTX 5070 Ti 16 GB, Ryzen 5 3600XT, 15 GB RAM in WSL).

Read these first, in order:

1. `docs/architecture.md`: what runs today, parameters, files, open problems.
2. `docs/analysis/quality-journey.md`: every path tried, the evidence that closed it, the decisions, and the incidents.
3. `docs/analysis/emotion-cloning.md`: the original emotion and voice-cloning plan.

## Goal and constraints

- **Goal:** natural Italian speech from a **small** model (today 6.8M parameters, about 27 MB fp32) that runs on CPU and in the browser, with emotion control (7 labels × intensity) and, later, voice cloning.
- **Italian only:** no English training datasets. The pretrained Vocos, trained on English, is allowed only as a training-time teacher or reference, never shipped.
- **Small model:** prefer distillation, fine-tuning and better training over bigger networks; keep latency close to today's.
- **GPU:** one job at a time, at most **85% utilisation and 250 W** (`scripts/gpu_governor.py`; WSL can't set the card's power limit). Two GPU jobs at once have crashed training.
- **How the user judges:** by ear. Send separate, loudness-matched WAV files, and keep the playground (`scripts/playground_server.py`, port 8790) on the newest checkpoints. The user reaches it through an SSH tunnel to `localhost:8790`, because the Mac's browser can't reach the LAN address.
- **Docs:** keep `docs/analysis/quality-journey.md` updated.
- **Commits:** only when asked, and **never** add Claude attribution (no Co-Authored-By, no Claude-Session, no "Generated with Claude Code").

## Current pipeline

text → espeak phonemes → **voice model** (`runs/voice-emotion/inference.pt`, step 103,292, 4.54M) → durations, pitch, voicing, energy → source-filter heads render an intermediate pulse-train waveform → **Vocos log-mel** (100 bands, hop 256) → **harmonic-prior vocoder** (`runs/vocoder-prior/vocoder-0150000.pt`, 2.23M; phase from a pulse train at the predicted pitch, network predicts magnitudes plus a phase correction) → 24 kHz audio.

## The problem now: the voice is metallic

The vocoder is no longer the main limit. The user says our prior vocoder is "pretty nice", close to Vocos. **The voice model's spectrogram carries the metallic trace.** The evidence comes from copy-synthesis on held-out test recordings (IT1 clip `cml-it-1595_3627_000311-0004`, CML4974 clip `cml-it-4974_5281_000561-0002`):

| Test | Input → vocoder | User's verdict |
|---|---|---|
| a | original recording | reference |
| b | **real** spectrogram → Vocos | "almost not metallic" |
| c | real spectrogram → our plain vocoder | "very metallic" |
| e | real spectrogram → our Vocos-distilled vocoder | "metallic" |
| g | real spectrogram → our **prior** vocoder (148–150k) | "pretty nice for now" |
| d | **voice model** spectrogram, with the recording's real pitch and timing → Vocos | "still metallic" |
| — | normal synthesis (predicted pitch and timing) → Vocos or our vocoder | "not great", similar to each other |

So a good vocoder on our model's spectrogram still sounds metallic. The likely causes:

1. **The intermediate pulse-train waveform.** It draws overly regular harmonics into the spectrogram. On the original DSP output, pulse-locking (how identically each band repeats every pitch cycle) at 1–2 / 2–4 / 4–6 kHz was 0.46 / 0.37 / 0.23 against 0.36 / 0.32 / 0.21 for the real voice, and the harmonic-to-noise ratio was too high at 1–4.5 kHz.
2. **Over-smoothing.** The voice model is trained with average-error losses only, without adversarial training, so its spectrogram is "average".
3. **Limits of size and data:** 4.54M parameters, about 55 h of mostly LibriVox audiobook audio. IT1 has almost no energy above about 10 kHz.

## What was tried and failed

All of it is in `quality-journey.md`, with numbers.

1. **Fixing the pulse-train (DSP) vocoder:**
   - cap the harmonics at 5 or 8 kHz;
   - pitch jitter of 1, 3 and 6%: the harmonic-to-noise ratio matched the real voice, but it was still buzzy;
   - a learned 2D STFT post-filter (about 55k params): pulse-locking got worse;
   - a frame-rate neural "spectral head" (1.1M) correcting the DSP spectrum: the user heard it as worse;
   - a stronger adversarial weight: no better.

   **Lesson:** average-error losses reward the perfectly regular waveform, and short fine-tunes can't fix a GAN vocoder.
2. **Phase 1, a direct mel output, with the pretrained Vocos voicing it.** It never receives exact pitch: its harmonic structure was 0.55 against 0.95 for the real recording.
3. **Plain Vocos-style vocoder (2.1M, our own) to 100k:** the user heard it as metallic. Pulse-locking was **too low** (0.19 / 0.14 / 0.06): incoherent phase, smeared harmonics.
4. **Distilling Vocos.**
   - The waveform-level complex-STFT loss stayed flat at about 0.487, the level of unrelated phases.
   - The frame-level spectrum loss (log magnitude + 1 − cos phase) learned the magnitude (0.75 → 0.42) but not the phase (1.00 → 0.991). The CPU check that "worked" had only memorised one batch.
   - Verdict: metallic.
5. **Pilot replacing the intermediate waveform with a direct spectrogram output** (5k steps, `runs/voice-mel-pilot`): the user said it was **worse**. The direct spectrogram is too smooth, and the vocoder had never seen it.
6. **First emotion data pilot** (VoiceDesign references on a neutral sentence, then Base cloning): no audible emotion. The references mostly shifted pitch level, with no loudness, rate or contour change.

## What worked

1. **More IT1 data plus a 6-voice voice model** (4 CML speakers with bad audio excluded):
   - the overfitting of it-v2 was removed;
   - pitch spread rose from 0.91 to 1.00 with the pitch-spread loss weight at 4, fixing the "monotone" measure;
   - the vocoder-input loss is in.
2. **Harmonic-prior vocoder:**
   - pulse-locking 0.53 / 0.59 / 0.53, close to the real 0.45 / 0.42 / 0.25 (plain vocoder: 0.19 / 0.14 / 0.06);
   - the discriminators no longer dominate (loss 0.24 instead of 0.06);
   - the user calls it "pretty nice".
3. **Synthetic emotion data:** Qwen3-TTS **VoiceDesign** with **English** style instructions on **emotion-congruent Italian sentences** gives "perfect emotions" by ear.
   - 2,280 clips accepted (`data/emotion-synth`), 6 designed voices, 7 emotions × 2 strengths;
   - 1,749 survive alignment (the aligner's quality checks drop the most expressive whispered or breathy clips).
4. **Emotion input** (intensity × learned vector):
   - works strongly for the synthetic voices;
   - carries over to IT1 only partly: pitch for joy and anger (151 → about 205 Hz at intensity 1.5); timing and the other emotions barely change.

## Open options for the metallic voice (not yet tried)

- **Adversarial or feature-matching loss on the voice model's spectrogram** (or its intermediate waveform), against over-smoothing.
- **Make the intermediate waveform less regular,** or have the vocoder rely less on it: more pair training, a higher pair fraction, longer training.
- **A direct spectrogram route that actually works:** the direct output plus anti-smoothing plus vocoder fine-tuning on it. Plain direct was worse.
- **Speaker dropout** for better emotion transfer to IT1; more voices or data against the growing train/validation gap.

## Useful commands and files

- **Resume the paused vocoder (150k → 184k):** `( setsid nohup scripts/training/alternate.sh >> logs/alternate.log 2>&1 < /dev/null & )`. Adapt the paths: the originals live in the git-ignored `logs/`.
- **Pulse-locking and harmonic-to-noise measurements:** see the code snippets described in quality-journey §2 and §6.1.
- **Copy-synthesis test clips:** the IDs above, from `data/it-multi-features/test.jsonl`.
- **Vocoder pairs:** `python -m tiny_voca.vocoder_pairs` (`data/vocoder-pairs/train`, 24,345 clips).
- **Tests:** `.venv/bin/python -m unittest discover -s tests` (84 tests, 2 skipped).
