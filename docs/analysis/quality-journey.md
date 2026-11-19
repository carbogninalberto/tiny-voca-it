# From "muffled and metallic" to a neural vocoder: what we tried and why

Working log of the quality work on the Italian fork, 16–17 November 2026. It
records the paths explored, the evidence that closed each one, the choices made,
and what is running now. Companion to `emotion-cloning.md` (the original plan).

Every number here was measured on this machine unless it is marked **[E]**
(estimate). "Validation" always means clips the model never trained on.

## 1. Starting point

The user's verdict on the it-v2 model (IT1 voice, CML-TTS speaker 1595): "muffled,
metallic/buzzy, robotic and monotone".

What the it-v2 numbers showed:

- **It was overfitting.** Train mel error was 0.40 against 0.61 on validation. The
  best validation was 0.496 at steps 6–8k. 100k steps was about 230 passes over
  11.4 h of audio.
- **Averaging checkpoints did not help.** Averaging the 60k–100k checkpoints
  gave the same 0.6087.

Step 1 of the plan addressed the data side:

- **All of IT1's audio:** 18.5 h of training audio instead of 11.4 h, with the same
  validation and test clips as it-v2.
- **Training changes:** a cosine learning rate and weight averaging (EMA).

It reached a validation mel error of **0.479 at 8k steps**, better than it-v2's
best, before the adversarial phase. Step 1 is paused at 14k (`runs/it1-full`) and
became the starting point for everything below.

## 2. Finding the real cause of the metallic sound

The model generated audio with a built-in DSP vocoder: a band-limited pulse train
plus noise, shaped by smooth spectral filters (`SourceFilter` in
`src/tiny_voca/model.py`). It is cheap (the whole model runs at about 0.02× real
time on one CPU thread) and gives exact pitch control.

**Copy-synthesis test.** We fitted the vocoder's inputs directly to a real IT1
recording, so it got perfect pitch and the best possible spectral settings. That
fit still sounded "very buzzy and metallic" to the user. The design itself was the
ceiling, not the training.

**What exactly is wrong.** We compared model and recording per frequency band,
time-aligned, with true pitch and timing:

| | 1–2 kHz | 2–4 kHz | 4–6 kHz |
|---|---|---|---|
| Pulse-locking¹, real voice | 0.36 | 0.32 | 0.21 |
| Pulse-locking, model | 0.46 | 0.37 | 0.23 |
| Pulse-locking, vocoder fitted to real | 0.49 | 0.41 | 0.34 |
| Peakiness², real / model | 3.7 / 3.5 | 4.8 / 3.9 | 4.5 / 3.9 |

¹ How identically each band's energy repeats every pitch period (autocorrelation at
the pitch lag). ² Waveform crest factor.

The low end (below 1 kHz) matched. Between 1 and 6 kHz, the real voice produces
sharp, slightly different bursts each pitch cycle. The pulse train produces
smoother but mechanically identical ones, and that is what the ear hears as buzz.
The harmonic-to-noise ratio was also too high in the same bands (for example 3.9 dB
against 1.2 dB at 2–3 kHz).

## 3. Fixes tried on the DSP vocoder, all rejected

| Attempt | Result | Why rejected |
|---|---|---|
| Cap the harmonics at 5 or 8 kHz, noise above | no audible change | the excess is at 1–6 kHz, not above |
| Pitch jitter in the pulse train (1%, 3%, 6%) | at 6% the harmonic-to-noise ratio matched the real voice | still buzzy; pulse-locking at 1–2 kHz stayed 0.47 |
| Learned 2D STFT post-filter (Wavehax-inspired, about 55k params) | slightly less buzzy at first | measured pulse-locking got **worse** (0.52 at 1–2 kHz); about 2× CPU time |
| Frame-rate neural "spectral head" correcting the DSP output's spectrum (Vocos-style, 1.1M params) | user: "worse" | pulse-locking up to 0.51 / 0.42 / 0.30 |
| Same head with a stronger adversarial weight (0.7 → 1.5) | no better | |

**Why none of this worked.** The spectral losses reward the *average* waveform. In
1–6 kHz the real voice varies randomly cycle to cycle, and the loss-minimising
prediction of random variation is a perfectly regular average. So more capacity
moved the output towards *more* regularity. Only adversarial training counters
that, and adversarial vocoders train for about 1 million steps, not 2–6k.

The jitter, post-filter and head code stays in the repository, off by default
(`excitation_jitter`, `postfilter_*`, `head_*` in `Config`). The failed runs were
deleted.

## 4. The decisive test: a real neural vocoder

We downloaded the pretrained Vocos 24 kHz vocoder (MIT license, 13.5M params,
trained on English audiobooks) and ran two tests:

- **A — real IT1 recording through Vocos:** "perfect".
- **B, C — our model's output, re-voiced by Vocos:** "way better, the buzz is very
  subtle now".

So the 24 kHz data was fine, and our model's predicted spectrum was largely fine.
The waveform stage was the problem. Decision: **replace the DSP vocoder with a
neural one.**

## 5. Second opinion, and a path we dropped (Phase 1)

**Phase 1 idea.** The model would predict Vocos' mel spectrogram directly, with a
new linear output, and the pretrained Vocos would voice it.

An independent review (Fable 5.2 at maximum effort, with all findings in its
brief) found a flaw before we listened. The linear output never receives exact
pitch, so it can't place the harmonics. Measured harmonic structure below 1.5 kHz:

| Mel source | Harmonic structure |
|---|---|
| Real recording | 0.95 |
| DSP output → mel (the path of tests B/C) | 0.90 |
| Phase 1 mel output | 0.55 |

We stopped Phase 1. Kept: the exact Vocos mel extractor (`VocosMel` in
`src/tiny_voca/losses.py`, matches Vocos to 2e-7) and the pipeline of tests B/C:
**voice model → DSP waveform → mel → neural vocoder**. The DSP stage now only
carries exact pitch and timing into the mel; the neural vocoder produces the sound.

The review's other findings:

- **40/48 kHz output is pointless with this data.** IT1 is already about 26 dB down
  at 11–12 kHz (LibriVox, MP3-sourced). No suitable full-band Italian corpus with a
  usable license is known. Decision: stay at 24 kHz.
- **Four step-2 speakers have bad audio:** CML659 (cut at about 7 kHz), CML10446
  (almost no highs), CML2019 (rolls off at about 8 kHz), CML1725 (about 20 dB SNR).
  They are excluded everywhere below.
- **The monotone is a prosody problem**, not a vocoder or sample-rate one.
- **The pretrained Vocos is English-trained:** fine as a reference, but we should
  ship our own Italian vocoder.

## 6. Our own neural vocoder (Phase 2)

| | |
|---|---|
| Code | `src/tiny_voca/vocoder.py`, `src/tiny_voca/train_vocoder.py` |
| Run | `runs/vocoder` |
| Architecture | Vocos-style: ConvNeXt blocks over mel frames and an inverse-STFT head; 192 channels, 8 layers, **2.1M params** (Vocos: 13.5M) |
| Input | the Vocos 100-band log-mel, so the pretrained Vocos stays a like-for-like reference |
| Data | 23,592 clips of 6 clean speakers: IT1 + CML4974, CML4998, CML6807, CML2033, CML1989; no transcripts needed |
| Training | random 0.68 s crops, batch 32, mel L1 × 45, the multi-period + multi-resolution discriminators (least squares), feature matching × 2; 300k steps, cosine LR |
| Speed | about 3.5–3.7 steps/s under the 230 W cap |

Batch 32 instead of 16 roughly doubled the throughput: 118 against 70 crops per
second. At batch 16 the GPU was waiting on kernel launches.

**Overfitting check.** The validation set is 315 IT1 test clips; none of their IDs
or sentences is in the training data. Copy-synthesis mel error (lower is better):

| Step | 5k | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|---|
| Train | 0.300 | 0.224 | 0.177 | 0.160 | 0.153 | 0.144 |
| Validation | 0.284 | 0.200 | 0.165 | 0.161 | 0.142 | 0.133 |

Both fall together, with no gap. Checkpoints are kept at 30k, 50k, 80k and 100k
(`runs/vocoder/keep-*`), each with the full training state and the inference weights.

| Step | 60k | 70k | 80k | 85k | 90k |
|---|---|---|---|---|---|
| Train | 0.140 | 0.138 | 0.131 | 0.129 | 0.127 |
| Validation | 0.131 | 0.129 | 0.131 | 0.121 | 0.121 |

**Batch 64 from 78k.** User limits: GPU at most 85% and 250 W.

- **Steps get slower but carry more audio:** 2,000 steps take 14 instead of 9 minutes,
  but the vocoder now learns from 148 instead of 115 crops per second (+29%).
- **A temporary bump:** validation rose for one check (0.129 → 0.131) and then
  reached its best (0.121).
- **Governor fix:** the governor overshot after load spikes (0.18 s pause per step,
  1.5 steps/s). It now backs off faster when well below target and raises the pause
  more gently (`scripts/gpu_governor.py`).

### 6.1 Still metallic: our vocoder is the limit

At 92k the user still heard a metallic trace. The training charts showed why:

- **The discriminators keep winning:** their loss fell from 0.30 to 0.07, while the
  vocoder's adversarial loss rose from 0.55 to 0.89.
- **Feature matching kept worsening:** 0.065 → 0.118, faster since about 55k.
- **The spectrogram error was still improving:** 0.177 at 20k → 0.125.

Copy-synthesis of real test recordings (no TTS involved) settled it. The user's
verdicts:

| | Vocos (13.5M) | Ours (2.1M, 94k) |
|---|---|---|
| Real recording → spectrogram → vocoder | **b: "almost not metallic"** | **c: "very metallic"** |

A phase-aware distance shows our vocoder, Vocos and the real recording all about
0.43 apart. The phase of a real recording can't be recovered from a mel
spectrogram, so each vocoder invents its own. Vocos' invented phase is clean and,
importantly, is a fixed function of its input, so our vocoder can learn to copy it.

**First attempt failed, then fixed.** The first distillation loss compared the two
**waveforms** with multi-resolution complex STFTs. It stayed flat at about 0.487 for
6,000 steps, the level of unrelated phases (run kept as
`runs/vocoder-distill-wave-loss`). Re-analysing waveforms with other window sizes
makes phase a bumpy target, and the gradients average out.

The fix compares the two **predicted spectra** directly. Our vocoder and Vocos
predict log magnitude and phase on the same 1024/256 STFT grid before their inverse
STFT, so they can be matched frame by frame (`spectrum_distance`): an L1 on log
magnitude plus a magnitude-weighted 1 − cos(phase difference), as in the APNet
vocoders. On a CPU check its phase term fell from 0.99 to 0.64 in 150 steps, and the
waveform-level distance from 0.38 to 0.26. The distillation was restarted from 100k
with this loss (`--distill-mode spectrum`, weight 4) at 17:46.

**Decision: distillation, no size increase.** Run `runs/vocoder-distill`:

- **Model and start:** the same 2.1M model, starting from the plain run at 100k.
- **Losses:** the usual ones plus an L1 between magnitude-compressed complex STFTs
  of ours and of Vocos' output on the same input (`complex_distance`, weight 10;
  it pulls the phase, not only the magnitude).
- **Length:** 100k → 160k steps.
- **Vocos' role:** a training-time teacher only; the shipped vocoder stays our
  Italian-trained 2.1M model.
- **The plain run:** stopped at 100k. Its checkpoint is the fallback.

**Distillation outcome.** At 110k the magnitude had converged on Vocos (distance
0.75 → 0.42; validation 0.112, later 0.108), but the phase term only went 1.00 →
0.991. Across varied data, Vocos' phase is too complex for a 2.1M network to
imitate; the CPU check had only memorised one batch. The user heard **e**
(distilled, 110k) as "metallic".

**Pulse-locking tells which way each vocoder is wrong** (IT1 test clip, 1–2 / 2–4 /
4–6 kHz):

| Version | Pulse-locking |
|---|---|
| a: original | 0.45 / 0.42 / 0.25 |
| b: Vocos | 0.45 / 0.36 / 0.21 |
| c: our plain vocoder | **0.19 / 0.14 / 0.06**, too irregular: phase not coherent from frame to frame, harmonics smeared (metallic) |
| e: distilled | 0.21 / 0.11 / 0.09, same |
| f: pulse-train prior, untrained | **0.69 / 0.68 / 0.70**, too regular (the old buzz) |

### 6.1b Harmonic-prior vocoder (`--prior`, run `runs/vocoder-prior`)

The vocoder builds a band-limited pulse train from the frame pitch (voiced frames)
plus a 0.03 noise floor (`Vocoder.analyse_prior`). The STFT phase of that signal is
the base of the output phase. The network gets the prior's log magnitude as a
second input (a 1×1 conv) and predicts each bin's log magnitude plus a **phase
correction**.

- **Size:** 2.22M (+0.1M).
- **Start:** warm start from the distilled vocoder at 142k; the phase rows start at
  zero, so the run starts from the prior's phase.
- **Training:** a fresh learning-rate schedule (`--fresh-schedule`, 2e-4, 1k
  warm-up, cosine over 42k steps).
- **Distillation:** magnitude only; phase now comes from the prior.
- **Pairs mixed in:** the fine-tune on the voice model's own spectrograms (§6.2) is
  part of this run (`--pairs`, 50%), not a separate stage.
- **Target:** pulse-locking near the real voice's, between c (metallic) and f
  (buzzy).
- **Started:** 00:07 on 18 November.

The playground feeds the voice model's predicted pitch and voicing to this vocoder.
The user, through the playground at about 146k: our vocoder is now "similar to
Vocos, a bit worse". The voice model's spectrogram, not the vocoder, became the
limit.

**Paused at 150k by request (01:50, 18 November):** the user found it "pretty nice
for now".

- **Kept:** `runs/vocoder-prior/vocoder-0150000.pt`, plus `last.pt` for resuming
  (`logs/alternate.sh` resumes it).
- **Validation:** 0.211 at 145k, 0.203 at 150k (real and our own spectrograms mixed,
  so not comparable with the plain runs).
- **Discriminators:** no longer dominate (loss about 0.24, against 0.06 before).
- **Pulse-locking on real recordings** (1–2 / 2–4 / 4–6 kHz):

| Clip | Prior vocoder, 150k | Real voice | Plain vocoder |
|---|---|---|---|
| IT1 | 0.53 / 0.59 / 0.53 | 0.45 / 0.42 / 0.25 | 0.19 / 0.14 / 0.06 |
| CML4974 | 0.53 / 0.53 / 0.49 | 0.50 / 0.43 / 0.33 | 0.17 / 0.14 / 0.06 |
| CML6807 | 0.60 / 0.62 / 0.56 | — | — |

It is now slightly on the regular side, much closer to the real voice than the
metallic plain vocoder.

### 6.3 Where the metallic trace lives (blind listening, 18 November)

All tests were blind and used two aligned test clips (IT1 and CML4974, with the
recording's pitch and timing) voiced by the frozen prior vocoder at 150k.

- **Band swaps:**
  - model + real 1–6 kHz: not clean;
  - model + real **above 1 kHz**: best hybrid ("clean, almost; clear");
  - model + real below 1 kHz: unchanged.

  So the trace is above 1 kHz.
- **Envelope vs texture:**
  - the real fine texture helped IT1;
  - the real envelope helped CML4974 a little.

  Both matter, depending on the voice.
- **Residual mel correction** (`MelCorrection`, 29k params, L1, 5k and 15k steps):
  error above 1 kHz 0.599 → 0.577 → 0.572; **no audible change**. An average-error
  objective can't create irregular texture.
- **Capping the intermediate pulse train** at 6, 4 or 2.5 kHz: no better (IT1
  slightly worse). The regularity of the high harmonics is not what is heard.
- **Spectrogram comparison** (real vs model, active frames, IT1):
  - below 1 kHz the detail matches (0.61 / 0.58);
  - above 1 kHz the model is **airbrushed**: fine detail across bands is 0.28 / 0.18
    at 1–3 kHz, 0.21 / 0.06 at 3–6 kHz and 0.33 / 0.15 at 6–12 kHz; frame-to-frame
    change is 25–35% lower.
  - Also visible: an edge at about 11.3 kHz (the pulse-train limit), and pulses
    leaking into silence.
- **Voicing gate without retraining:** the /s/, /f/, /ʃ/ hisses vanished. **The
  model had learned to build them from the pulse train** (a periodic "hiss")
  instead of from noise.
  - A soft gate keeping 50% or 25% of the pulse energy in unvoiced frames, with a
    25 ms fade, was judged "some good".

**Round 3** (`logs/round3.sh`, started about 03:30, 18 November):

1. **Adversarial pilot:** the existing waveform discriminators on the voice model,
   103k → 113k (`runs/voice-adv-pilot`).
2. **Soft-gated fine-tune:** new `Config.unvoiced_pulse_gain` = 0.5, using recorded
   voicing in training and predicted voicing at synthesis (exported to ONNX too),
   with light adversarial training (ratio 0.5), 113k → 128k (`runs/voice-gate`).
3. **New vocoder pairs** from the gated model, built on the GPU
   (`data/vocoder-pairs-gate`).
4. **Prior vocoder resumed** on them, 150k → 180k (`runs/vocoder-prior2`).

Blind comparison files are built at the end of each stage.

**Progress (08:10, 18 November):**

- **Adversarial pilot:** done at 03:42. The spectrogram error rose slightly, 0.743 →
  0.757, as expected when trading average accuracy for realism.
- **Gated fine-tune:** done at 05:05, error 0.750 (no cost from the gate).
- **Pairs:** rebuilt on the GPU in 15 minutes.
- **Vocoder:** the error on real test recordings keeps falling: 0.203 at 150k, 0.192
  at 165k.
- **Browser preview:** the gated model with the 168k vocoder is published on GitHub
  Pages.
- **Listening verdicts** on the A and G files are still pending.

**Blind verdicts:**

- **A (adversarial pilot):** the real spectrogram was rated "good" for both voices.
  - IT1: 103k "metallic" → adversarial 113k "a bit metallic".
  - CML4974: both "metallic".
  - The sentences were "almost identical".
  - So: a small gain, on one voice only.
- **V (vocoder 150k vs 180k):**

  | Input | Vocoder 150k | Vocoder 180k |
  |---|---|---|
  | Real IT1 | good (very subtle trace) | good |
  | Real CML4974 | a bit metallic | good |
  | Gated model IT1 | a bit metallic | metallic |
  | Gated model CML4974 | metallic | metallic |
  | Gated model, sentence | a bit metallic | metallic |

  - The extra 30k steps on the gated model's pairs made the vocoder **better on real
    spectrograms and worse on the model's**. It renders the model's smooth spectrogram
    more literally rather than hiding it.
  - **Objective check:** 85 test clips of the model's own spectrograms, with
    teacher-forced timing and pitch, measured against the real recording.

    | Vocoder | Mel error > 1 kHz | Mel error < 1 kHz | Fine detail > 1 kHz (real = 1) | Pulse locking 1–2 / 2–4 / 4–6 kHz (real 0.53 / 0.47 / 0.40) |
    |---|---|---|---|---|
    | 150k | 0.822 | 0.773 | 0.82 | 0.69 / 0.63 / 0.58 |
    | 180k | **0.806** | **0.751** | 0.78 | **0.63 / 0.56 / 0.45** |

    - 180k is closer to the real voice on spectrogram error and pulse locking (the
      buzz measure), and slightly smoother.
    - The user asked to follow the data, so **180k is kept** in the playground and on
      Pages. (It was briefly switched back to 150k after the listening test.)
  - **Both vocoders lock to the pulses more than real speech does** (0.58–0.45 vs
    0.40 at 4–6 kHz). Real spectrograms through either vocoder lock the same way,
    0.60 at 4–6 kHz, so this residual buzz is in the vocoder, not only in the voice
    model.
- **G (gated fine-tune), blind, through vocoder 150k:**

  | Voice | Real | Adversarial 113k | Gated 128k |
  |---|---|---|---|
  | IT1 | good | subtle metallic, still good | a bit metallic |
  | CML4974 | a bit metallic | metallic | metallic |

  The sentences were both "metallic".

  Objective check (85 test clips, vocoder 180k):

  | Voice model | Mel error > 1 kHz | Mel error < 1 kHz | Fine detail > 1 kHz | Pulse locking |
  |---|---|---|---|---|
  | 103k | 0.805 | 0.758 | 0.74 | 0.61 / 0.55 / 0.45 |
  | Adversarial 113k | **0.802** | **0.749** | **0.78** | 0.64 / 0.56 / 0.46 |
  | Gated 128k | 0.806 | 0.751 | 0.78 | 0.63 / 0.56 / 0.45 |

  - Adversarial training restored some fine detail (0.74 → 0.78).
  - The soft gate adds nothing measurable, and by ear it is slightly worse on IT1.
  - **Round 4 starts from the adversarial model (113k), without the gate.**

**Pilot: direct spectrogram instead of the DSP route (rejected).**

- **Setup:** the voice model got a 100-band Vocos-mel output (`vocos_mels=100`),
  trained 5k steps from 103k (`runs/voice-mel-pilot`). The same weights then voiced
  3 sentences two ways with the prior vocoder at 148k:
  - via the old DSP waveform → mel ("PRIMA");
  - via the direct mel ("DOPO").
- **Result:** the user heard "DOPO seems worse".
- **Why:** a directly regressed mel is over-smoothed and undertrained, and the
  vocoder never trained on it, while the DSP route keeps harmonic detail from the
  pitch.
- **Decision:** keep the DSP route. Improve the vocoder on that route's own
  spectrograms (the pairs in the prior run). Revisit direct prediction only together
  with an anti-smoothing (adversarial) loss.

**Incident:** stopping the watchdog at 01:06 also stopped the switching script and
the vocoder, because the watchdog had restarted them as its own child processes.
The vocoder resumed from 148k at 01:31. All launches now double-fork, so jobs are
re-parented away from whoever started them.

### 6.2 The voice model's spectrogram carries a trace too

**Test d:** the voice model was given the recording's real pitch and timing, and
its spectrogram was voiced with Vocos. The user: **"still metallic"**, while b (the
real spectrogram, same vocoder) is not.

So the voice model's internal pulse-train stage leaves a trace in the spectrogram
that any vocoder reproduces. Training time won't remove it.

**Decision: fine-tune the vocoder on our own spectrograms,** the standard
two-stage-TTS remedy.

- **Pairs** (`tiny_voca.vocoder_pairs`): the final voice model runs every training
  clip with the clip's real pitch and timing, so its spectrogram lines up with the
  recording frame for frame. The spectrogram is stored as float16, about 24k clips
  in `data/vocoder-pairs/train`.
- **Fine-tune** (`train_vocoder --pairs`, run `runs/vocoder-ft`, 160k → 190k):
  - 70% of paired crops use our spectrogram as input, the rest the real one; the
    target is always the recording;
  - the distillation target is Vocos on the **real** spectrogram (clean, like b),
    never on ours.

## 7. Voice model, block 1: better data, tuned for the new pipeline

| | |
|---|---|
| Run | `runs/voice` (14k → 64k) |
| Data | the same 6 clean voices, 22,596 clips (`data/it-multi6-annotations`, selection in `selection.json`); IT1 stays the main voice |
| Start | branched from step 1 at 14k, with the 5 voices appended |
| Changes | adversarial phase removed (the neural vocoder makes the final sound; about 2× faster); **new loss on the vocoder's actual input**, the Vocos mel of the DSP output (`vocos_mel_loss`, weight 10); pitch-spread weight 2 → 4 against the monotone |

Results, validation:

| | Start (14k) | End (64k) |
|---|---|---|
| Vocoder-input error | 0.807 | **0.739** |
| Pitch spread (1.0 = as varied as the real voices) | 0.91 | **1.00** |
| Pitch error | about 2.0 semitones | 2.7 semitones |

The monotone measure is fixed. The higher pitch error is the expected cost: a
lively curve that differs from the recording differs by more than a flat one.
We'll keep watching it. A move from discrete pitch choices to sampling them,
against the remaining flatness, is still planned (`emotion-cloning.md`).

## 8. Emotion

**What does not exist.** There is no Italian emotional speech at our quality.

**First pilot, failed.** The original plan (`emotion-cloning.md` §6.2):

1. VoiceDesign speaks an emotional reference clip, using a neutral sentence.
2. Base clones that clip on ordinary sentences.

The user judged the 14 sample clips: not recognisable, not emotional. Measurement
showed why: the references themselves mostly changed pitch level (the male voice
went from 99 Hz neutral to 164 Hz "joy"). Loudness, rate and pitch movement barely
changed, and anger was not louder than neutral. Cloning then copied only that.

**Re-test, worked.** Emotion-congruent sentences ("Basta! Non ne posso più!") with
strong instructions, generated with VoiceDesign directly. The user, comparing
English and Italian instructions: English gives "perfect emotions".

**Generated data.** 6 designed voices × (neutral + 6 emotions × 2 strengths) × 30
sentences, screened with Whisper small (Italian, 20% word-error threshold):

| | |
|---|---|
| Accepted | **2,280 of 2,340 clips** (`data/emotion-synth`) |
| Generator | `scripts/generate_emotions.py`, sentences in `data/emotion-texts.json` |
| Split | by sentence: 1,896 train / 231 validation / 153 test; no validation or test sentence appears in training in any voice or emotion |
| License | Apache-2.0 weights; the audio is synthetic and must be labelled as such |

**Real reference.** Emozionalmente (CC BY 4.0, Zenodo) was downloaded: 6,902 clips
from 431 amateur actors. 3,418 clips have at least 80% of raters recognising the
intended emotion and good audio. It is 16 kHz on mixed microphones with only 18
sentences, so it serves as the yardstick for emotion, not as training data for now.

**Model.** New `Config.emotions` (7 labels). Each utterance adds intensity × a
learned vector, in the same places as the speaker vector. Neutral or intensity 0
adds nothing (tested to give exactly the old output), so all recorded corpora
train unchanged. Emotion is separate from the speaker, so IT1 can be asked for
"anger 0.7" without ever having recorded anger. **Whether that transfer works is
the main open risk.**

**Processing.** Features plus phoneme alignment, on the CPU while the GPU trained:

| Split | Accepted | Rejected by the aligner |
|---|---|---|
| Train | 1,749 | 147 |
| Validation | 222 | 9 |
| Test | 145 | 8 |

Most rejections are vowels without detected voicing (81) and over-long phones (46),
mostly disgust, sadness and fear: whispered, breathy or drawn-out delivery. So the
aligner's quality checks remove some of the most expressive clips. They can be
relaxed for this data if those emotions come out weak.

**Voice block 2** (`runs/voice-emotion`, 64k → 114k, started 15:06):

- **Data:** 24,345 clips, the 6 recorded voices plus the 6 synthetic emotional
  voices (12 voices).
- **Emotion input:** 7 emotions.
- **Loading:** the dataset loader now combines several annotation manifests
  (`--train a.jsonl b.jsonl`), each checked against its own annotation run.

**First result at 87k** (same sentence, intensity 1.0, voiced with Vocos):

| | Neutral | Joy | Sadness | Anger |
|---|---|---|---|---|
| IT1 pitch / duration | 149 Hz / 4.01 s | 171 / 4.05 | 144 / 4.03 | 171 / 4.04 |
| EF2 (synthetic) pitch / duration | 269 Hz / 4.29 s | 364 / 4.32 | 256 / 4.34 | 394 / 3.94 |

The synthetic voices change strongly. IT1, which never recorded an emotion, gets
only a small pitch shift and no timing change: the cross-speaker transfer is weak so
far. Options if it stays weak:

- **Intensity above 1** at synthesis.
- **Speaker dropout** in training, so the emotion vector can't encode "synthetic
  voice".
- **More synthetic voices.**
- **Emotional IT1-like data** made by voice conversion.

**Overfitting watch** (validation − training on the vocoder-input error):

| Point | Gap |
|---|---|
| Block 1, 46k | 0.04 |
| Block 1, 64k (end) | 0.06 |
| Block 2, 85–91k | about 0.09 |

**The early stop ended block 2 at 103k.** Validation was best at 95k (0.735) and
flat or slightly worse for the next 8 checks (0.737–0.740). The final model
(`runs/voice-emotion/inference.pt`, 103k, EMA) is 0.5% behind the best checkpoint,
which is negligible. At 103k IT1's emotion transfer is unchanged: at intensity 1.5,
joy goes to 203 Hz and anger to 208 Hz against 151 Hz neutral; sadness, fear and
surprise barely move, and timing doesn't change. For
the next voice training: more data (cleaned-up excluded speakers, more synthetic
voices) or stronger regularisation.

## 9. How the machine is run

- **One GPU job at a time.** `logs/alternate.sh` switches the GPU between blocks:
  1. vocoder to 50k
  2. voice to 64k
  3. emotion data
  4. vocoder to 100k
  5. voice block 2 (emotion)
  6. distilled vocoder (100k → 160k), with the pairs built on the CPU meanwhile
  7. fine-tune on the voice model's spectrograms (160k → 190k)

  How it behaves:
  - **Early stop:** a voice block stops if validation hasn't improved for 8 checks
    (8,000 steps).
  - **No silent skipping:** it refuses to continue if a block ends early without a
    stop request.
  - **Resumable:** it skips blocks already done, and doesn't start a second copy of
    a run that is already training, so it can be restarted at any time.
  - **Process matching:** it recognises processes only by their exact
    `python -m tiny_voca…` command line.
- **GPU limits:** `scripts/gpu_governor.py` pauses training between steps to hold the
  user's caps: first 230 W, now **85% utilisation and 250 W**. The card's maximum is
  300 W, and WSL can't change its power limit.
- **Watchdog:** `logs/watchdog.sh`, every 2 minutes.
  - It restarts the switching script or the playground if they die.
  - It alerts on no progress for 20 minutes, GPU memory above 15.5 GB for 6 minutes,
    any new crash in the logs, or Windows C: below 20 GB free.
- **Playground:** `scripts/playground_server.py`, port 8790.
  - It always serves the newest voice and vocoder checkpoints and shows their step
    counts.
  - Vocoder choice: ours, the old DSP one, or Vocos for reference.
  - Controls: speed, intonation, 20 test sentences, and emotion and intensity once
    the emotion model exists.
  - LAN access needs a Windows port forward, like the existing playground on 8765.

## 10. Incidents and lessons

| What happened | Cause | Change |
|---|---|---|
| Two trainings at once crawled at 0.06 and 0.12 steps/s | 16 GB GPU memory full, WSL spilled to system memory | one GPU job at a time |
| Post-filter run died silently at 19:02 | not confirmed; most likely killed for memory | jobs started detached (`setsid`) with their own logs |
| Voice crashed at 04:03 with a CUDA "illegal memory access" | emotion generation running alongside, GPU memory near full | generation became its own block; watchdog alerts on full memory |
| Voice block stopped after exactly 1 hour | `tiny_voca.train` has a default `--max-seconds 3600` | `--max-seconds 172800` in the block scripts; the switching script checks the step reached |
| First watchdog alert was a false alarm | two bugs in the watchdog script | fixed and tested |
| Reported emotion processing as started before it was | my error | corrected the same morning |
| The vocoder's launch script kept an old governor (80%, 300 W) after a resume | separate scripts had diverged | one governor setting per launch script, updated together |
| Process lookups matched my own shell command (twice): once the switching script waited for a vocoder that wasn't running, once a command stopped itself | `pgrep -f` on a pattern that also appeared in the command text | lookups match only `python -m tiny_voca…` in the program position |
| Governor overshoot cut the vocoder to 1.5 steps/s | slow recovery after a load spike | faster back-off below target |

## 11. Choices at a glance

| Option | Chosen? | Reason |
|---|---|---|
| Keep the DSP vocoder, fix it | no | fitted to real audio it is still buzzy; five fixes failed (§3) |
| Pretrained Vocos as the shipped vocoder | interim reference only | English-trained, 13.5M params, about 2× our CPU cost |
| Model predicts the mel directly (Phase 1) | no | no exact pitch, harmonics smeared (§5) |
| **DSP waveform → mel → own small neural vocoder** | **yes** | the path the user rated "way better" (B/C); exact pitch kept; 2.1M params |
| 40/48 kHz output | no | the data has no content above about 10 kHz |
| Multi-speaker voice model with 6 clean voices | yes | more data and intonation variety; 4 bad speakers dropped |
| Adversarial phase on the voice model | no | the neural vocoder makes the final sound; 2× faster without |
| Emotion via cloning emotional references | no | the user heard no emotion; prosody barely changed |
| **Emotion via VoiceDesign, emotional text, English instructions** | **yes** | the user: "perfect emotions" |
| Emozionalmente as training data | not yet | 16 kHz, mixed microphones, 18 sentences; used as the reference |
| Vocoder batch 32 → 64 | yes | +29% audio per second within the 85% / 250 W caps |
| Keep the plain vocoder to 300k | no | plateaued on a metallic residue that copy-synthesis pins on the vocoder (c vs b) |
| Bigger vocoder (about 3.7M) | not yet | the user wants the model small; distillation first |
| **Distil the 2.1M vocoder from Vocos** | **yes** | Vocos' phase is learnable from the mel; no size increase |
| **Fine-tune the vocoder on the voice model's own spectrograms** | **yes** | test d: the voice model's spectrogram carries the trace even through Vocos |

## 12. Open items

- **Voice block 2 with emotion:** check whether IT1 takes on emotion it never
  recorded (weak at 87k), against the real Emozionalmente clips.
- **Distilled vocoder (100k → 160k), then fine-tune on our spectrograms
  (160k → 190k).**
  - Distillation is judged on copy-synthesis of real recordings: c should approach b.
  - The fine-tune is judged on the full TTS pipeline: d should approach b.
- **ONNX export of the neural vocoder,** and a real single-thread latency
  measurement for the browser.
  - **[E]** total cost about 1.3–1.6× today's CPU time.
  - **Measured on 2 threads in PyTorch:** 9.6 s of speech in 0.29 s, voice model
    plus vocoder.
- **Italian text normalisation:** "sig.ra" → "signora", "via XX Settembre" →
  "venti", "12,50 euro" → "dodici euro e cinquanta", "email" → "imeil".
- **Discrete pitch with sampling,** if the remaining flatness bothers the user.
- **Commit the work.** Nothing is committed yet.

### 6.4 Round 4 (18 November, from about 10:05)

**Voice model:**

- Adversarial 113k → 128k with the new **mel critic**
  (`losses.MelDiscriminator`, 41k parameters, training only). It judges the Vocos
  log-mel above 1 kHz, which is what the vocoder reads and where the model
  over-smooths.
- The adversarial share is raised to 1.0 (from 0.7). No pulse gate.
- Run `runs/voice-mel-critic`, script `scripts/training/voice-mel-critic.sh`.

**Vocoder:** pulse share of the prior weakened at high frequencies
(`VocoderConfig.prior_high_hz` / `prior_high_gain`, no extra parameters, exported
to ONNX).

- **Sweep without retraining** (vocoder 180k, real spectrograms, 30 test clips; pulse
  locking per band 1–2 / 2–4 / 4–6 kHz; real speech 0.55 / 0.48 / 0.41):

  | Setting | Pulse locking | Mel error > 1 kHz |
  |---|---|---|
  | Off | 0.68 / 0.64 / 0.63 | 0.150 |
  | 2.5 kHz, gain 0.3 | 0.71 / 0.63 / 0.60 | 0.183 |
  | 2.5 kHz, gain 0.1 | 0.72 / 0.62 / 0.54 | 0.230 |
  | 2 kHz, gain 0.03 | 0.73 / 0.56 / 0.29 | 0.259 |

  The setting controls high-band locking as intended. The mel error rises until the
  network adapts.
- **Fine-tune:** 180k → 195k at 2.5 kHz / gain 0.05 (`runs/vocoder-highband`),
  queued after the voice run.

**Success criteria** (objective, on the model's own speech):

- fine detail above 1 kHz towards 1.0 (now 0.78);
- pulse locking at 4–6 kHz towards 0.41 (now 0.45);
- mel error not worse;
- then blind listening.

### 6.5 Two voices (decided 18 November)

The final model needs only two voices: **IT1** (male) and **EF3** (female; chosen by
timbre).

**Per-voice measurement** (adversarial 113k + vocoder 180k, 10 test clips each,
error above 1 kHz):

| Voice | Vocoder on the real spectrogram | Voice model's spectrogram |
|---|---|---|
| IT1 | 0.148 | 0.658 |
| CML6807 | 0.154 | 0.649 |
| CML1989 | 0.144 | 0.721 |
| CML4998 | 0.146 | 0.750 |
| CML2033 | 0.159 | 0.926 |
| CML4974 | 0.149 | 0.968 |

- **The vocoder is voice-independent:** about 0.15 error for every voice.
- **The voice model is much worse on some voices,** and not because of data amount:
  CML4974 has 13 h of training data and is the worst.
- Twelve voices dilute a 4.5M model. Two voices concentrate it.

**Candidate female voices:**

- CML6807: rejected, strong Sicilian accent.
- No other large female CML-TTS reader exists.
- Public Italian female sets are 16 kHz (M-AILABS) or commercial.

**EF3 data:**

- EF3 had only 23 minutes, each clip designed separately, so the timbre drifts.
- **`scripts/generate_voice.py`** clones one EF3 clip (`emo-EF3-neutral-0.0-3e06518d08`,
  picked by ear) with Qwen3-TTS Base on about 6 h of CML-TTS sentences, screened by
  Whisper.
- The audio is synthetic and labelled as such.

**Round 5** (`scripts/training/round5.sh`):

1. EF3 data, features and alignment.
2. Fine-tune on IT1 + EF3 + the emotion set, 20k steps, adversarial ratio 0.7, no mel
   critic, from adversarial 113k. Round 4 scored 0.804 / 0.75 detail vs 0.802 / 0.78.
3. Vocoder 180k → 200k on IT1 + EF3, with the high-band prior.

The planned 15k-step high-band vocoder run on all voices (§6.4) is replaced by step 3.

### 6.6 Decision log (round 5, 18 November)

From now on decisions are made and recorded here, without waiting for approval.

- **Round 4 base: adversarial 113k.**
  - The mel-critic run scored 0.804 / detail 0.75 vs 0.802 / 0.78. The mel critic
    stays in the code, off.
  - A review notes the test was confounded: one critic of nine, averaged, on short
    crops. So this is not evidence that mel critics fail.
- **EF3 generation throughput.**
  - The Qwen3-TTS clone loop is bound by one CPU core: a nested HF `generate()` per
    codec frame, with the GPU at 10–30%.
  - Benchmark, audio seconds per wall second:

    | Batch | 2 | 4 | 8 | 16 | 24 |
    |---|---|---|---|---|---|
    | Speed | 0.56 | 0.95 | 1.57 | 3.08 | 3.33 |

  - Two processes only time-slice the GPU, and three ran out of memory.
  - Default length cap of 2,048 frames: one runaway clip held a batch for minutes.
  - **Chosen:** one process, batch 20, length cap 2 × expected + 3 s, batches sorted
    by length, Whisper screening on the GPU in a worker thread overlapped with the
    next batch.
  - Whisper on CPU int8 was slower than generation and stalled it.
  - Result: 26 clips/min (about 2.2× real time), up from 4–12. Target 6 h.
  - **GPU memory.** With `expandable_segments` the process grew to 15.7 of 16 GB and
    spilled into shared memory; the rate fell to about 14 clips/min and the watchdog
    alerted.
  - **Fix:** cap the process at 80% of GPU memory (`--memory-fraction 0.8`) and use
    batch 18. It now runs stable at 12.4 GB and about 21 clips/min.
- **EF3 splits:**
  - **Leakage fixed.** Sentences that are training sentences of another voice stay in
    train (`check_leakage` would have aborted the fine-tune).
  - **Validation sentences.** Validation sentences of the other voices are added to
    the pool and routed to EF3 validation, giving a per-voice validation set.
  - **Training gates:** WER ≤ 0.1 and 11–17 characters/s. The manifest keeps
    everything screened.
- **Emotion set:** the VoiceDesign EF3 *neutral* clips (drifting timbre, about 3 dB
  louder) are dropped from training, so the clone defines EF3's neutral voice. The
  emotional EF3 clips stay.
- **Voice recipe:**
  - Kept the 113k recipe: adversarial ratio 0.7, feature matching 30. Only the data
    changes.
  - The review notes that feature matching dominates the GAN term and that the ratio
    excludes the ×10 Vocos-mel loss. To revisit once the two-voice result is in.
- **Vocoder:**
  - mel weight 15 instead of 45. The review notes the critics are averaged over 8
    networks, about 8× weaker than Vocos's sum, a plausible cause of the high-band
    over-locking.
  - This is combined with the high-band prior change and the removal of paired
    training, so the effects are confounded. It is judged on pulse locking and blind
    listening.
- **Durable libraries:** `qwen_tts`, `vocos` and friends are copied to
  `~/.local/share/tiny-voca-libs` (`EXTRA_LIBS_DIR`), out of the temporary
  scratchpad.
- **Texture test** (CPU, while the GPU generates). Hypothesis: point-wise L1 on log
  spectra punishes noise texture, so the model airbrushes the highs. Test on 3 IT1
  clips: real / model / model + real-statistics texture above 2 kHz / model
  unsharp-masked above 1 kHz. Blind files sent; if texture helps, the next voice
  round changes the loss above 2 kHz.
- **EF3 data result (14:47):**
  - 6.03 h passed screening (3610 clips, Whisper WER ≤ 0.2).
  - The training gates kept 2336 clips: 2224 train, 98 validation and 14 test, or
    3.62 h of train audio. WER above 0.1 dropped 1118 clips; the speaking-rate
    gate dropped another 156.
  - Decision: keep the strict 0.1 gate. Clean data is worth more than quantity for
    a fine-tune that starts from an existing model, and 3.6 h is ~45% of IT1's
    training audio. If EF3 comes out mispronouncing words or unstable, generate
    more rather than loosen the gate.
- **Post-production pilot (16:15):**
  - Setup: blind test on the voice-two checkpoint at step 122,790, IT1 + EF3, all
    variants at the same loudness. The variants were the raw output, a de-harsh
    EQ, multiband compression, and multiband + full-band compression + small room
    reverb.
  - Result: the listener picked P2, which was the **raw** output for both voices.
  - Decision: no post-processing in the output path. Static EQ and dynamics do not
    remove the metallic trace; they only colour the voice. The fix stays in the
    models (the vocoder high-band prior, the voice loss above 2 kHz).
  - Why lower playback volume sounds less metallic: masking at the ear, not a
    property of the audio.
- **Voice-two result (17:01, step 133,292):**
  - Setup: test clips scored with the model's own spectrograms through the 180k
    vocoder; the reference model is voice-adv-pilot.
  - **EF3** (10 clips): mel error above 1 kHz 0.849 → **0.570** (−33%); fine detail
    0.85 → 0.88; pulse locking at 4–6 kHz 0.46 → 0.57 (real speech 0.50).
  - **IT1** (40 clips): mel error above 1 kHz 0.638 → 0.638 (unchanged); detail
    0.78 → 0.78; locking 0.44 → 0.47 (real 0.31; the vocoder is mostly
    responsible, and the vocoder-two retrain addresses it).
  - Decision: adopt voice-two. EF3 gains a lot and IT1 holds. The browser preview
    and the playground now use it with the 180k vocoder (Pages `4a4b3b4`). Blind
    old/new files were sent; key in `voicetwo-key.json`.
- **Voice-two blind test and checkpoint choice (17:40):**
  - Listener: old better for IT1, new better for EF3.
  - Checkpoint sweep (EMA weights, 40 IT1 + 14 EF3 test clips; mel error above
    1 kHz / pulse locking at 4–6 kHz; real IT1 0.31, real EF3 0.48):

    | Checkpoint | IT1 mel | IT1 lock | EF3 mel | EF3 lock |
    |---|---|---|---|---|
    | voice-adv-pilot (old) | 0.638 | 0.44 | 0.837 | 0.46 |
    | 115k | **0.635** | **0.44** | 0.584 | **0.55** |
    | 120k | 0.632 | 0.46 | 0.572 | 0.59 |
    | 125k | 0.643 | 0.47 | 0.570 | 0.58 |
    | 130k | 0.639 | 0.47 | 0.583 | 0.58 |
    | 133k (final) | 0.638 | 0.47 | 0.582 | 0.57 |

  - Ear and data agree: IT1's locking creeps up during the fine-tune and the ear
    hears it as more metallic.
  - Decision: use **step 115k** (`runs/voice-two/inference-115000.pt`). IT1 is
    unchanged, EF3 keeps almost all of its gain, and EF3 locks the least of the
    new checkpoints. Applied to the playground and Pages (`1784209`).
  - Next voice round: the locking creep points at the adversarial term
    sharpening harmonics above 4 kHz. To do: a shorter fine-tune, or the
    noise-friendly loss above 2 kHz.
- **Vocoder-two result (19:54, 200k):**
  - Setup: voice model 115k; 40 IT1 + 14 EF3 test clips. Old = vocoder-prior2
    at 180k. Columns: mel error above 1 kHz / fine detail (real = 1) / pulse
    locking at 4–6 kHz; real locking is 0.31 for IT1 and 0.48 for EF3.

    | | IT1, real mel | IT1, model mel | EF3, real mel | EF3, model mel |
    |---|---|---|---|---|
    | old 180k | 0.148 / – / 0.62 | 0.635 / 0.78 / 0.44 | 0.163 / – / 0.64 | 0.584 / 0.88 / 0.55 |
    | vocoder-two 200k | 0.157 / – / 0.45 | 0.662 / 0.85 / 0.45 | 0.168 / – / 0.47 | 0.601 / 0.91 / 0.50 |

  - The 190k–200k checkpoints measure the same (converged early).
  - The high-band prior did its job on real spectrograms: locking fell
    0.62 → 0.45 for IT1, and EF3 now matches real speech.
  - On model spectrograms, detail improves and EF3 locking falls, but IT1
    locking does not move: IT1's remaining lock comes from the voice model's
    spectrogram, not the vocoder. Mel error rises 3–4%, as expected from the
    lower mel weight.
  - Decision: no objective winner on both voices, so the blind test (W files,
    key `vocoder-two-key.json`) decides. Until then the preview keeps the 180k
    vocoder; the playground offers both.
- **Vocoder-two blind test (20:15):**
  - Result: the listener preferred **vocoder-two 200k** on both voices (IT1-W2,
    EF3-W1).
  - The ear follows the locking and detail numbers, not the 3–4% higher mel
    error.
  - Decision: the current product is voice-two 115k + vocoder-two 200k, in the
    playground and on Pages (`a61b882`).
  - Lesson: for the metallic trace, pulse locking and fine detail are better
    predictors of preference than mel error.
- **Round 6 diagnosis (20:40): the remaining IT1 lock is in the vocoder.**
  - This corrects the earlier note. With vocoder-two, IT1 locks the same on real
    and on model spectrograms (0.45 / 0.45; real speech 0.31), so the voice model
    is not adding to it.
  - Inference-only sweep of the high-band pulse share (`prior_high_gain`, trained
    at 0.05) and its start frequency. Columns are model-spectrogram mel error /
    pulse locking at 4–6 kHz:

    | gain, from | IT1 | EF3 |
    |---|---|---|
    | 0.05, 2.5 kHz (trained) | 0.662 / 0.45 | 0.601 / 0.50 |
    | 0.02, 2.5 kHz | 0.663 / 0.42 | 0.603 / 0.42 |
    | 0.0, 2.5 kHz | 0.665 / 0.38 | 0.605 / 0.35 |
    | 0.05, 1.5 kHz | 0.675 / 0.36 | 0.614 / 0.42 |
    | 0.0, 1.5 kHz | 0.677 / 0.12 | 0.612 / 0.17 |

  - Locking follows the gain almost linearly, at almost no cost in mel error at
    2.5 kHz. Below 2.5 kHz the error rises (outside what the vocoder was trained
    on).
  - Decision: the gain becomes a per-voice setting chosen blind (G files, key
    `gain-key.json`). If the best gain differs per voice, store it per voice in
    the model metadata. If the gain is far from 0.05, fine-tune the vocoder with
    the gain randomised in [0, 0.1] so every setting is in distribution. This
    replaces the planned voice-loss round for now.
- **Per-voice pulse share (22:20):**
  - Blind G files: the listener heard no difference between gains 0, 0.02 and
    0.05 on either voice ("all very good").
  - Decision: per the data, take the gain that brings each voice's 4–6 kHz
    locking closest to its real speech. **IT1: 0.0** (0.38 vs real 0.31).
    **EF3: 0.05**, the trained value (0.50 vs 0.48).
  - Implementation: `VocoderConfig.voice_high_gain`, a per-voice override.
    `Vocoder.forward(..., voice=name)` uses it, and `infer` and the playground
    pass the voice. The ONNX graph holds a per-voice gain table indexed by the
    `voice` input, so the browser applies it with no new inputs.
  - Tested: export parity and config round trip (`test_per_voice_high_gain`);
    89 tests.
  - File: `runs/vocoder-two/vocoder-voicegain.pt`, live on Pages (`7542565`)
    and the default in the playground.
- **Playback level and two-voice product (22:30):**
  - Why my samples sounded better than the playground: the local playground
    peak-normalised to 0.8, about 3 dB louder than the −20 LUFS samples, and
    louder playback makes the metallic trace more audible. It also applied the
    "neutral" emotion at 0.7 by default.
  - Fix: every path now plays at the same level (RMS 0.1, about −20 LUFS, peaks
    ≤ 0.95). That covers CLI samples, the local playground and Pages. The emotion
    intensity defaults to 0.
  - Product voices: only **IT1** and **EF3** are offered (the user's choice).
    The model keeps its 12 speaker rows, a negligible cost; Pages lists only the
    preview model, with `voices` = IT1, EF3. The playground uses `--voices IT1
    EF3`. Pages `1aaee9c`.

### 6.7 Round 6 (18 November, evening)

- **Vocoder-three:** a fine-tune from vocoder-two (200k) to 210k, with the
  high-band pulse share drawn per utterance from [0, 0.1] (`--prior-gain-range`).
  - Why: IT1's chosen setting (0.0) was outside the training distribution
    (trained at 0.05 only). The aim is to close IT1's remaining locking gap
    (0.38 vs real 0.31) without moving EF3.
  - Same data, losses and distillation as vocoder-two; learning rate 5e-5,
    checkpoints every 1k steps.
  - It will be judged per voice on model spectrograms with the per-voice gains,
    plus a small sweep of IT1 gain below 0.05.
- **EF3 "tremble" (23:10):**
  - Report: the listener heard EF3 trembling at times.
  - Measured on 12–14 test sentences, predicted pitch vs the recordings:

    | | frame-to-frame jump (st) | wobble (st) |
    |---|---|---|
    | EF3 model | 0.372 | 0.409 |
    | EF3 real | 0.278 | 0.287 |
    | IT1 model | 0.376 | 0.384 |
    | IT1 real | 0.311 | 0.334 |

    Wobble is the residual of a 9-frame moving average.
  - Spectral balance is not the cause: EF3 is within ±0.4 dB of the real clips
    in every band except −2.3 dB at 4–6 kHz.
  - Probable cause: the pitch-spread loss (weight 4) is satisfied partly by
    frame-level wiggle rather than intonation. The predicted spread is still
    below real (EF3 2.0 vs 2.8 st).
  - Fix at inference time: `TinyVoca.smooth_pitch`, a 5-frame Hann low-pass of
    log F0 that restores the voiced spread. It is in PyTorch and in the ONNX
    graph (`PITCH_SMOOTH_FRAMES`).
  - After the fix: EF3 0.268 / 0.322, IT1 0.274 / 0.296. Blind S files sent; key
    `smooth-key.json`.
  - Next voice round: add a pitch-delta loss (match the frame-to-frame change to
    the recordings) so the model learns intonation, not jitter. Also consider
    pitch_range around 1.2, since spread is about 70% of real.
- **Tremble blind test:** the listener preferred the smoothed version on both
  voices (EF3-S2, IT1-S1). Pitch smoothing is now the default everywhere: CLI,
  playground, and Pages (`23e1624`).

### 6.8 Benchmark and landing page (19 November, 00:15)

- **Vocoder-three adopted:**
  - IT1 at gain 0: real-spectrogram mel error 0.172 → 0.159, model-spectrogram
    error 0.665 → 0.663.
  - EF3: unchanged.
- **Public benchmark:** MiniMax TTS Multilingual Test Set, Italian split, scored with
  Whisper-large-v3.
  - Result: **WER 0.80%, CER 0.51%** (IT1 1.00%, EF3 0.66%). The published
    systems range from 0.95% (Qwen3-TTS 1.7B) to 2.16%.
  - These are not identical tasks: they clone Common Voice speakers zero-shot.
    See `docs/benchmarks/minimax-it/`.
- **Speed:** 22× real time on one CPU thread; 79× on an RTX 5070 Ti at batch 1.
- **Landing page:**
  - A dark style modelled on kuayle.com. It has a playground with autoplay and a
    waveform that follows playback and stops with it.
  - Sections: architecture, benchmark table, notes, and a roadmap with the build
    timestamp.
  - The final roadmap item is **tiny-voca-it-zs**: zero-shot cloning from 15 s,
    in a separate, bigger model (about 8–10M) with its own voice-fingerprint
    network.
  - Pages `0cac58a`.

### 6.9 Round 8: new product voices and stronger emotions (19 November, 00:45)

- **Voices:** the user chose two of ten VoiceDesign proposals. Designs are in
  `data/voice-refs/designs-20261119.json`.
  - **EM4** (design M4): a man in his thirties with a soft, gentle, medium-low
    voice, Tuscany; median F0 97 Hz.
  - **EF4** (design F3): a woman in her forties with a low, smooth news-anchor
    voice, Milan; median F0 215 Hz.
  - They replace IT1 and EF3 as product voices. IT1 is a real reader with no
    emotional recordings, so it cannot carry the emotion control.
- **Neutral data:** 5 h screened per voice, Base clones of each voice's neutral
  reference. Splits are leakage-safe across all voices (new `--split-train` /
  `--split-validation` flags).
- **Emotions were too weak; the cause is data.** Only 30 texts per emotion, so
  about 25 for training, and about 35 min per voice.
  - New texts: about 120 per emotion (`data/emotion-texts-v2.json`), written to
    be emotion-congruent.
  - New audio: about 100 clips per voice × emotion × intensity.
  - A WavLM speaker-verification filter against the reference removes clips
    whose VoiceDesign timbre drifted.
  - Emotion data is oversampled in training. An emotion classifier plus
    per-emotion pitch, energy and rate statistics will judge it.
- **Folded in:** the paused round 7 (pitch-change loss, weight 10) becomes part of
  this voice fine-tune.
- **Display names:** EM4 is shown as **Lorenzo** and EF4 as **Chiara** on the page,
  the playground and in the README. The internal speaker IDs stay EM4/EF4, so the
  running, resumable data stages are not disturbed.
- **Emotion texts v2:** 150 per emotion and 105 neutral (the old 30 per emotion
  plus 120 new). The new sentences are emotion-congruent, 7–22 words, no digits,
  and checked against duplicates. Copy: `docs/data/emotion-texts-v2.json`.
- **Emotion baseline** (`scripts/evaluate_emotion.py`): voice-two 115k on EF3,
  6 sentences per emotion, measured relative to neutral. Columns: pitch shift
  (semitones) / range change / rate ratio.

  | emotion | data | model |
  |---|---|---|
  | joy | +13.7 / +4.8 / 0.84 | +10.2 / +4.9 / 0.94 |
  | sadness | +5.1 / −1.4 / 0.76 | −0.1 / +0.2 / 0.91 |
  | anger | +14.8 / +3.1 / 1.06 | +11.1 / +3.4 / 1.02 |
  | fear | +5.4 / +0.2 / 0.94 | +5.0 / +3.0 / 0.95 |
  | surprise | +6.1 / +3.8 / 0.84 | +6.1 / +5.1 / 0.86 |
  | disgust | +5.0 / +1.0 / 0.80 | +0.7 / +1.5 / 0.94 |

  - The pitch shifts of joy, anger and surprise come through.
  - Sadness and disgust barely move.
  - Tempo changes (sad and disgusted speech is slower) and loudness changes are
    mostly lost.
  - This is the target for round 8.
- **Faster data generation (01:15):**
  - Profile: the per-frame Hugging Face `generate()` of Qwen3-TTS's code
    predictor (15 sub-codes per codec frame) takes 52% of the time. Almost all
    of it is kernel-launch and Python overhead; the GPU is about 15% busy.
  - `scripts/qwen_fast.py` replaces it with a lean loop: one cache and the same
    sampling rule (temperature, top-k, top-p).
  - Measured on 18 EM4 sentences: 3.24× → **3.88× real time (+20%)**, with
    identical Whisper WER (0.029). Now on in both generators.
  - A CUDA-graph capture of the whole sub-code loop failed: a device-side assert
    on the second replay, from state carried in the static cache. It would
    roughly halve generation time and is worth revisiting off the critical path.
  - Neither CPU generation (about 0.1× real time) nor a second GPU process
    (measured earlier: time-slicing, no gain) helps. Instead, idle cores run
    each voice's features and alignment while the GPU generates the next voice.
- **CUDA graphs adopted (01:55):**
  - The earlier failure was a lifetime bug: the cache-position tensors read by
    the graph were freed after capture and their memory reused. Fixed by keeping
    them on the graph object. In greedy mode, graph and eager output are
    identical.
  - Controlled test: 72 EM4 sentences, production memory cap 0.8 and expandable
    segments.

    | | real-time factor | mean WER | clips with WER > 0.2 | peak memory |
    |---|---|---|---|---|
    | fast loop | 5.01× | 0.097 | 10 | 11.8 GB |
    | CUDA graphs | **12.15×** | 0.091 | 4 | 11.8 GB |

  - Production now runs at about 50 accepted clips per minute (21 before).
  - Lesson: test speedups with the production settings (memory cap,
    allocator). The first uncapped test spilled into shared memory and looked
    like a hang.
- **Numbers as words (02:45):**
  - `tiny_voca.italian_numbers` (Python) and `playground/web/numbers.js`
    (browser) expand numbers before phonemisation: cardinals with Italian
    elisions, decimals, %, € $ £ amounts, times, dates, ordinals, signs, and
    long digit strings read digit by digit.
  - The two use ASCII regex semantics so they behave identically.
    `tests/test_numbers.py` checks parity through node.
  - The step sits outside the front-end identity, so checkpoints are unaffected.
  - Live on Pages for the current model. The benchmark's 11 digit lines (1.93%
    WER) will be re-measured with the round-8 model.
- **Round-8 fixes (02:50):**
  - CPU prep rejected every EM4 clip as an unknown speaker. Fix:
    `configs/italian-voice-round-8.json`, the emotion config with EM4 and EF4
    appended. Branching supports appended voices; existing rows keep their
    positions.
  - New training option `--new-voice-init EM4=EM1 EF4=EF3`: appended voices
    start from the speaker rows of a similar existing voice rather than random
    rows.
- **Emotion stage memory spill (04:23):**
  - The watchdog alerted: VoiceDesign at batch 16 with no cap reached 15.8 GB
    and spilled into shared memory (7 clips in 8 minutes).
  - Fix: `generate_emotions.py --memory-fraction`, run at 0.8 with expandable
    segments, the same settings as the neutral stage.
  - Now 11 GB and about 32 accepted clips per minute; done around 05:35.
- **Prep result (05:34):**
  - The voice-similarity filter kept 1079 EM4 and 1083 EF4 emotional clips and
    dropped 194 each. Median similarity to the reference was 0.89–0.95 for
    every emotion (EM4 is lowest for joy and anger, 0.893).
  - Emotion train split: 1858 clips (oversampled ×3 with unique IDs).
  - Neutral train: EM4 1679 clips, 2.92 h; EF4 1795 clips, 2.88 h.
  - Alignment probes: EF4 fricative→vowel boundaries are 98.6% within 2 frames.
- **Voice-four** started at 05:38 after one fix: duplicate IDs are not allowed,
  so oversampled copies get an `~k` suffix. It runs at 2.8 steps/s, so the
  20k steps finish around 07:40.
- **Voice-four result and first publish (08:10):**
  - Spectral error above 1 kHz (validation, model spectrograms through
    vocoder-three): EM4 0.569 → 0.549 and EF4 0.631 → 0.597, from 118k to 135k.
    It plateaued after about 130k.
  - Emotion mismatch: the mean over emotions of |Δpitch| + |Δrange| +
    |Δloudness| + 10·|Δrate| against the data. EM4 5.18 → 6.08, EF4 7.68 → 6.77.
    Step 135k is best overall.
  - Pitch cues are learned. **Tempo is not**: sadness is 0.93× in the model and
    0.80× in the data.
  - Fix: `model.EMOTION_SPEED`, a per-emotion speed multiplier at intensity 1,
    linear in intensity, in `synthesize` and in the ONNX graph. Values: sadness
    0.86, anger 1.08, surprise 1.06, fear 1.05. Only the ratios consistent across
    both voices are used; the pitch residuals disagree between voices and are
    left alone. Tested (`EmotionSpeedTests`).
  - Published voice-four 135k with vocoder-three to Pages, labelled
    "Lorenzo + Chiara". The playground offers EM4 and EF4.
  - On "double the data and train to 260k": the curve plateaued after about
    6k steps, and IT1 with 11.4 h plateaued at a similar error. The limit looks
    like model capacity, so a modestly larger model is the better experiment.
- **Vocoder-four and benchmark (10:30):**
  - Vocoder-four (225k, fine-tuned on EM4/EF4/emotion data plus IT1) beats
    vocoder-three on both voices. Validation, real/model mel error: EM4
    0.156/0.549 → 0.149/0.546, EF4 0.178/0.598 → 0.176/0.597.
  - Per-voice high-band gain: the value whose locking matches real speech. EM4
    0.0 gives 0.44 (real 0.44); EF4 0.05 gives 0.51 (real 0.52).
  - Published on Pages.
  - MiniMax Italian benchmark with Lorenzo and Chiara and numbers as words:
    **WER 0.54%** (95% 0.20–0.96), CER 0.33%, 90/100 lines perfect. Lines with
    digits: 1.93% → 0.32%. Previous IT1 + EF3 model: 0.80%.
  - A permanent watchdog (`logs/watchdog-main.sh`) now turns every round-log
    line into an event, alerts on tracebacks, GPU spill and RAM, restarts dead
    rounds and keeps the playground alive. Lesson: the round-8 watchdog exited
    with the round, and 30 minutes were lost after vocoder-four finished.
- **Release 0.2.0 published (10:55)** at the user's request: GitHub Releases v0.2.0 (latest).

### 6.10 Round 9: a deeper voice model (19 November, 10:30)

- **Why:** the round-8 curves plateaued after about 6k steps, and IT1 with 11.4 h
  plateaued at a similar error, which points to the model's capacity.
- **Change:** acoustic stack 5 → 8 blocks (`configs/italian-voice-round-9.json`).
  The voice model grows from 4.54M to 5.73M parameters, the total from 6.77M to
  about 7.96M.
- **Branch:** from the release weights (voice-four 135k with EMA weights in
  `step-00135000-ema.pt`), using `--identity-new-blocks`. The new pre-norm
  residual blocks start with zero output projections, so they pass input
  through unchanged. Verified: after one step at lr 1e-9, the output differs
  from the release by 2e-4 on a 0.55 peak. The first attempt differed by 0.36
  because it started from the raw training weights, not the EMA weights.
- **Training:** 20k steps on the round-8 data, scored every few thousand steps
  against the release with vocoder-four.
- **Kept only if** it is better on both voices (spectral error, locking, emotion
  match) and in a blind test. It is about 26% larger and somewhat slower on CPU.

### 6.11 Smaller ONNX files (19 November, 11:30)

- **Lean graph:** the STFT analysis and synthesis kernels (about 20 MB of float32,
  partly duplicated) are computed inside the graph from the windows
  (`export._runtime_kernels`, tied to a runtime zero so the exporter cannot
  fold them back). The angles use the exact integer k·n mod N.
  - 47.4 → **28.0 MB**. Export parity tests pass.
  - Speed: 26.7× → 24.4× real time on one CPU thread (tables rebuilt per
    sentence).
- **Variants** (`scripts/quantize_onnx.py`), checked with
  `scripts/compare_onnx.py` on 40 MiniMax sentences. The graph's noise is
  seeded, so fp32 vs fp32 is exactly 0.

  | variant | size | same durations | mel L1 to fp32 | Whisper-small WER |
  |---|---|---|---|---|
  | fp32 lean | 28.0 MB | – | 0 | 3.83% |
  | fp16 weights (fp32 compute) | 14.6 MB | 40/40 | 0.23 | 3.83% |
  | int8 dynamic (MatMul, Conv) | 8.2 MB | 2/40 | – (timing differs) | 3.72% |
  | int4 MatMulNBits, block 32 (rest fp16w) | 5.9 MB | 0/40 | – | 4.41% |

  - All variants run at 22–24× real time on one CPU thread.
  - int8 and int4 change the predicted durations of most sentences;
    intelligibility is the same for int8 and slightly lower for int4.
  - A blind test (Q files, key `quant-key.json`) decides which go to the page
    and the release.
- **Quantization decision (12:30):**
  - Blind listening: the user heard no difference between fp32, fp16, int8 and
    int4.
  - Decision: int8 is the browser default, following the data (int4 measured
    +0.6 WER points).
  - ONNX Runtime Web 1.20 has no `ConvInteger` kernel, so the browser int8 file
    quantizes MatMul only and stores the other large weights as fp16. It is
    8.6 MB. Whisper-small WER on 40 sentences: 3.95% against 3.72% for fp32,
    within noise.
  - Live on Pages: the first download drops from 47 MB to 8.6 MB.
- **Round 9 verdict (12:57):**
  - The deeper model (5.73M) at 155k: mel error EM4 0.541 and EF4 0.592,
    against the release's 0.545 and 0.597 (about −1%).
  - Emotion mismatch is mixed (EM4 worse, EF4 slightly better).
  - Decision: keep the 4.54M release model. The gain is too small for 26% more
    parameters under the keep-it-small constraint. Capacity is not the main
    bottleneck.
- **Intonation range is already right** for the new voices: predicted pitch
  spread is 104% of the data for EM4 and 100% for EF4. The "70%" figure
  belonged to IT1/EF3. No change.

### 6.12 Round 10: texture loss above 2 kHz (19 November, 13:00)

- **Loss:** `--texture`. For every Vocos mel band centred at or above 2 kHz, match
  the mean absolute frame-to-frame change per utterance to the recording's,
  rather than the values. This is the variability analogue of the pitch-spread
  loss. Model output has about 0.88× the recorded detail there; point-wise L1
  rewards the average.
- **Setup:** weight 20 (about 9% of the loss), 8k steps at lr 5e-5 from the
  release EMA weights. Judged on detail, mel error and locking per voice, plus
  a blind test.
