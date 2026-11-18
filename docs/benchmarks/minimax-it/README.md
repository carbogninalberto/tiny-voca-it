# Italian benchmark: MiniMax TTS Multilingual Test Set

The Italian split of [MiniMaxAI/TTS-Multilingual-Test-Set](https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set)
(100 sentences, CC BY-SA 4.0): `italian_female` lines are read by EF3 and `italian_male` lines by IT1.

- **Model:** voice-two step 115k, vocoder-three 210k (per-voice high-band pulse share), pitch smoothing 5 frames.
- **ASR:** faster-whisper large-v3, beam 5.
- **WER/CER:** lower case, punctuation removed (seed-tts-eval style).
- **Command:** `scripts/benchmark_minimax_it.py`. Per-line transcripts are in `scored.jsonl`.

| System | Params | WER % | SIM | Source |
|---|---|---|---|---|
| **tiny-voca-it (IT1 + EF3)** | **6.8M** | **0.80** | n/a | measured here |
| Qwen3-TTS 12Hz 1.7B Base | 1.7B | 0.95 | 0.817 | Qwen3-TTS report, Table 6 |
| Qwen3-TTS 25Hz 1.7B Base | 1.7B | 1.11 | 0.718 | Qwen3-TTS report, Table 6 |
| Qwen3-TTS 25Hz 0.6B Base | 0.6B | 1.17 | 0.722 | Qwen3-TTS report, Table 6 |
| Fish-Audio S2 | – | 1.27 | 0.747 | dots.tts report, Table 3 |
| MiniMax-Speech | – | 1.54 | 0.699 | MiniMax-Speech paper, Table 2 |
| VoxCPM 2 | – | 1.56 | 0.780 | dots.tts report, Table 3 |
| ElevenLabs Multilingual v2 | – | 1.74 | 0.579 | MiniMax-Speech paper, Table 2 |
| dots.tts (MF4) | 2B | 2.16 | 0.843 | dots.tts report, Table 3 |

**Detail for tiny-voca-it:**

- WER by voice: IT1 1.00%, EF3 0.66%.
- CER: 0.51%.
- The 11 lines with digits score 1.93%; the other lines score 0.65%. Most of the gap is digits written
  as words ("tre a uno" for "3 a 1"), not misreadings.
- 89 of the 100 lines are transcribed without an error.

**Speed:**

- 22× real time on one CPU thread.
- 79× real time on an RTX 5070 Ti at batch 1 (74 ms median per sentence, including eSpeak).

**Caveat: the task is not the same.** The other systems clone the two Common Voice reference speakers
zero-shot, which is harder than speaking in voices the model was trained on. Read the table as
intelligibility at a given size, not as a ranking of cloning quality. For the same reason, SIM does not
apply to tiny-voca-it.
