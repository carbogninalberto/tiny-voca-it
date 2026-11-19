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

## Also noted from SmollerTTS

- **Δmel loss** (L1 on frame-to-frame mel differences at all frequencies): related to our texture loss
  (above 2 kHz, matching the amount of change). It could be tried at all frequencies.
- **Three ONNX graphs** (encoder · frame decoder · vocoder) with the length regulator in JavaScript: the
  split true streaming needs.
- **UTMOS** as an automatic naturalness score, now measured for our model too
  (`docs/benchmarks/minimax-it/`).
