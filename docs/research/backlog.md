# Research backlog

Ideas queued for after the zero-shot work (tiny-voca-it-zs). Each one is kept only if it measures and sounds
at least as good as the current release. Results go to `docs/analysis/quality-journey.md`.

## 1. Convolution-only acoustic stack

**Source:** SmollerTTS (Dun Li Chan, 2026), a 2.81M English TTS model. Its decoder is six convolution-only
frame blocks: depthwise conv → LayerNorm → 160→320 → GELU → 320→160, with a residual scaled by a learned
per-channel α. It predicts the mel spectrogram from the length-regulated phonemes plus prosody.

**Idea:** replace our 5 transformer blocks in the acoustic stack (1.99M parameters, full self-attention
over all frames) with convolution-only blocks, and possibly the 2 prosody blocks as well.

**Why it might help**
- Smaller and faster. Attention over thousands of frames is the most expensive part of the voice model on
  CPU.
- **A local receptive field makes true streaming exact.** Frames can be rendered in chunks with a fixed
  context, and the output equals full-sentence synthesis. With full attention, chunking changes the output
  (see the streaming plan in the roadmap).

**Risks:** attention lets distant frames coordinate, for example sentence-level intonation and loudness.
Convolutions only see a window. The prosody stack (which predicts pitch) might need to keep attention while
the acoustic stack goes convolutional.

**Experiment**
1. Branch from the release with an acoustic stack of N convolution blocks (kernel 7, a few dilations to
   widen the receptive field) instead of attention. Distil from the release model's acoustic features
   first, then fine-tune with the usual losses.
2. Measure:
   - mel error above 1 kHz, detail and locking per voice;
   - WER on the MiniMax Italian set;
   - UTMOS;
   - CPU speed;
   - a blind test against the release.
3. If it holds, measure how big the chunk context must be for exact chunked output.

## 2. Pronunciation-override dictionary

**Source:** SmollerTTS lets users supply an explicit pronunciation, and splits compounds only where tested.

**Idea:** a small user dictionary, word → phonemes (with stress and geminates), applied before eSpeak NG, in
both the Python front end and the browser front end (parity-tested like `numbers.js`).

**Why:** eSpeak misreads some names, loanwords and acronyms ("tiny-voca" → *tinivòka*). A dictionary fixes
them exactly without retraining, and the page can let users add entries for their own text.

**Plan**
1. Format: `docs/lexicon/it.tsv` (word, phonemes in our symbol inventory, note). Case-insensitive whole-word
   match, applied after number expansion.
2. The front end inserts the given phonemes directly instead of calling eSpeak for that word. Keep it
   outside the front-end identity (like the number rules), so checkpoints keep their fingerprint.
3. Seed it with frequent problem words found by Whisper on the benchmark and on a list of common loanwords.
4. In the browser: an optional "pronunciation" box (word = how it should sound) stored locally.

## 3. Naturalness: where the UTMOS gap is (measured 19 November)

UTMOS22-strong on 30 validation clips per voice (Lorenzo / Chiara):

| | UTMOS |
|---|---|
| real clips (Qwen3-TTS training data) | 4.04 / 4.22 |
| real 100-band log-mel → Vocos 13.5M | 3.75 / 3.74 |
| real log-mel → our vocoder 2.2M | 3.47 / 3.56 |
| full model, text → speech (0.3.0) | 3.42 / 3.27 |

- The **mel representation itself** costs 0.3–0.5: even Vocos cannot recover the recording from it.
- **Our vocoder vs Vocos** costs 0.2–0.3.
- **The voice model** costs only about 0.05 (Lorenzo) and 0.3 (Chiara).

**Ideas, in order of expected gain per effort**
1. **A better vocoder at the same size:** longer training, stronger adversarial weight (the vocoder's GAN term
   is averaged over 8 critics, about 8× weaker than Vocos's sum), feature matching on real clips. Target: close
   most of the 0.2–0.3 to Vocos.
2. **Bypass the mel bottleneck:** condition the vocoder on the voice model's acoustic features (192 per frame)
   plus the pitch prior, and fine-tune the two jointly on waveform losses (VITS-style). This is the only route
   past the ~3.75 ceiling of any mel vocoder. It loses the Vocos teacher on identical input, so distil first
   and then fine-tune jointly.
3. **A modestly larger vocoder** (2.2M → 3–4M) only if 1 plateaus.

Judged on UTMOS, the copy-synthesis split above, WER and blind tests.

## Also noted from SmollerTTS

- **Δmel loss** (L1 on frame-to-frame mel differences at all frequencies): related to our texture loss
  (above 2 kHz, matching the amount of change). It could be tried at all frequencies.
- **Three ONNX graphs** (encoder · frame decoder · vocoder) with the length regulator in JavaScript: the
  split true streaming needs.
- **UTMOS** as an automatic naturalness score, now measured for our model too
  (`docs/benchmarks/minimax-it/`).

## 4. Swappable voice adapters (planned, NOT started; decided 20 November)

Blueprint: frozen multi-speaker base · per-voice adapter of ~110k numbers (conditional LayerNorm scale/shift in
every block ≈ 8k, LoRA rank 4–8 on the prosody and acoustic feed-forward layers ≈ 100k, explicit log-F0
median/spread and tempo) · frozen text encoder and shared vocoder · a hypernetwork (voice code → adapter) for
an instant zero-shot start, polished by a 1–3 min fine-tune on Qwen3-TTS speech in the target voice. Adapters
are ONNX inputs: swapping voice = swapping a 200 KB tensor. Sources: AdaSpeech / AdaSpeech 4, residual adapters
(Morioka et al. 2022), Mixture of Adapters, HyperTTS, Meta-TTS.

Steps, each gated on the previous measurement (ECAPA vs a recording session not used for fitting, pass ≥ 0.538):
1. **Adapter-size test (~1 h GPU):** Qwen3-TTS generates ~10 min in the user's voice; train adapters of
   ~10k / ~100k / ~300k numbers on the frozen Lorenzo base; score each. Question: does ~100k reach 0.54?
2. **Base with adapter slots (~2 days):** retrain the base with conditional LayerNorm driven by a speaker
   code, on Qwen3-TTS clones of 1,000+ real Italian speakers.
3. **Hypernetwork (~1 day):** map a whitened voice code (speaker embedding + F0/tempo/long-term spectrum)
   to the adapter; zero-shot start, optional short fine-tune.
