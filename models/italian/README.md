# tiny-voca Italian checkpoints

Two inference checkpoints (format 4, Italian language profile v1, 24 kHz). Each has
4,534,987 parameters and is 18.2 MB on disk, about 17 MB in GPU memory.

| file | voice | what it is | sha256 |
|---|---|---|---|
| `tiny-voca-it-IT1.pt` | `IT1` | Italian base, trained from scratch, step 90,000 of 100,000 (selected on validation) | `828e221e68825363cf1f0fc7b4f55cba467c37c0b1759dc5591e7b2f61f06420` |
| `tiny-voca-it-IT2-adapted.pt` | `IT2` | custom-voice adaptation of the base (acoustic mode, 6,000 steps, 28 min of a second speaker) | `38460d6bcc36fa92e99ce8c3a78f1a5c87de026e7c06b677e59cf89ec718eac8` |

The IT2 checkpoint was adapted from the step-100,000 base (`provenance.adapted_from`
inside the file records its hash).

## Use

Needs eSpeak NG 1.51 with the `it` voice on `PATH` (see `docs/italian.md`).

```sh
python -m tiny_voca.infer --checkpoint models/italian/tiny-voca-it-IT1.pt \
    --text "Buongiorno! Come stai?" --output ciao.wav
python scripts/playground.py --checkpoint IT1=models/italian/tiny-voca-it-IT1.pt \
    --checkpoint IT2=models/italian/tiny-voca-it-IT2-adapted.pt     # http://127.0.0.1:8765
```

## Results (details in `docs/italian.md`)

- IT1, locked test set (314 clips), faster-whisper `small`, language `it`: WER 7.1%,
  CER 1.6%; the original recordings score 9.1% / 2.8% with the same ASR.
- IT2: test WER 13.0% (recordings 15.2%, 14 clips); mean pitch 100 Hz vs 107 Hz in its
  recordings (the IT1 base is about 150 Hz).
- Known limits: long sonorants (`ll nn mm rr`) are not reliably distinguished from
  short ones; no native-listener evaluation yet; no Italian ONNX export yet.

## Data and licence

Trained on CML-TTS Italian (Oliveira, F. S. et al., "CML-TTS: A Multilingual Dataset
for Speech Synthesis in Low-Resource Languages", 2023), CC BY 4.0, derived from
LibriVox public-domain audiobook recordings, via the Hugging Face mirror
`ylacombe/cml-tts` at revision `48aedd7a8ea08efc75cae3e81fbefa9720df0812`. IT1 was
trained on CML-TTS speaker 1595 (11.4 h) and IT2 on speaker 4998 (28 min). The
voices belong to those LibriVox readers; do not present them as anyone else's
voice. The weights are released under Apache-2.0 like the rest of the repository,
with the dataset attribution above.
