# Italian benchmark: MiniMax TTS Multilingual Test Set

The Italian split of [MiniMaxAI/TTS-Multilingual-Test-Set](https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set)
(100 sentences, CC BY-SA 4.0): `italian_female` lines are read by Chiara (EF4) and `italian_male` lines by Lorenzo (EM4).

- **Model (v0.3.0):** voice-six step 139k (texture loss), vocoder-four 225k (per-voice high-band pulse share), pitch smoothing 5 frames, numbers read as words.
- **Previous run:** IT1 + EF3 (round 5, no number normalisation), WER 0.80%, kept in `round5-it1-ef3/`.
- **ASR:** faster-whisper large-v3, beam 5.
- **WER/CER:** lower case, punctuation removed (seed-tts-eval style).
- **Command:** `scripts/benchmark_minimax_it.py`. Per-line transcripts are in `scored.jsonl`.

| System | Params | WER % | SIM | Source |
|---|---|---|---|---|
| **tiny-voca-it 0.3.0 (Lorenzo + Chiara)** | **6.8M** | **0.50** | n/a | measured here (v0.2.0: 0.54) |
| Qwen3-TTS 12Hz 1.7B Base | 1.7B | 0.95 | 0.817 | Qwen3-TTS report, Table 6 |
| Qwen3-TTS 25Hz 1.7B Base | 1.7B | 1.11 | 0.718 | Qwen3-TTS report, Table 6 |
| Qwen3-TTS 25Hz 0.6B Base | 0.6B | 1.17 | 0.722 | Qwen3-TTS report, Table 6 |
| Fish-Audio S2 | – | 1.27 | 0.747 | dots.tts report, Table 3 |
| MiniMax-Speech | – | 1.54 | 0.699 | MiniMax-Speech paper, Table 2 |
| VoxCPM 2 | – | 1.56 | 0.780 | dots.tts report, Table 3 |
| ElevenLabs Multilingual v2 | – | 1.74 | 0.579 | MiniMax-Speech paper, Table 2 |
| dots.tts (MF4) | 2B | 2.16 | 0.843 | dots.tts report, Table 3 |

**Detail for tiny-voca-it:**

- WER by voice: Lorenzo 0.27%, Chiara 0.66%.
- CER: 0.32%.
- v0.2.0 (before the texture loss): WER 0.54%, kept in `v0.2.0/`.
- The 11 lines with digits score 0.32%, down from 1.93% before numbers were read as words.
- 90 of the 100 lines are transcribed without an error.

**Speed:**

- 22× real time on one CPU thread.
- 79× real time on an RTX 5070 Ti at batch 1 (74 ms median per sentence, including eSpeak).

**Caveat: the task is not the same.** The other systems clone the two Common Voice reference speakers
zero-shot, which is harder than speaking in voices the model was trained on. Read the table as
intelligibility at a given size, not as a ranking of cloning quality. For the same reason, SIM does not
apply to tiny-voca-it.
