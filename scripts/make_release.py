"""Build a release bundle: voice model, vocoder, one-file ONNX, model card, samples, benchmark, checksums.

    python scripts/make_release.py --version 0.2.0 --voice runs/voice-four/inference.pt \
        --vocoder runs/vocoder-four/vocoder-voicegain.pt --voices EM4=Lorenzo EF4=Chiara \
        --benchmark docs/benchmarks/minimax-it/result.json [--draft]

Writes release/<version>/. With --draft it also creates a DRAFT GitHub release (gh CLI) with the
files attached; nothing is public until the draft is published by hand.
"""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SAMPLES = ["Buongiorno! Questa è una prova del modello tiny-voca, addestrato da zero sulla lingua italiana.",
           "Il treno per Firenze parte alle 12:30 dal binario 7.",
           "Hai già prenotato il ristorante per domani sera?"]
EMOTION_SAMPLES = {"joy": "Non ci posso credere, abbiamo vinto! È la notizia più bella dell'anno!",
                   "sadness": "Le sue scarpe sono ancora all'ingresso, e non ho il coraggio di spostarle.",
                   "anger": "Avevi promesso di venire, e invece sei rimasto in ufficio un'altra volta!"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", required=True); p.add_argument("--voice", required=True); p.add_argument("--vocoder", required=True)
    p.add_argument("--voices", nargs="+", required=True, help="ID=DisplayName for the product voices")
    p.add_argument("--benchmark", type=Path); p.add_argument("--draft", action="store_true")
    a = p.parse_args()
    from tiny_voca.export import export, metadata
    from tiny_voca.infer import load_model, match_frontend
    from tiny_voca.stream import Streamer
    from tiny_voca.vocoder import load_vocoder
    out = Path("release") / a.version; out.mkdir(parents=True, exist_ok=True)
    names = dict(v.split("=", 1) for v in a.voices)
    shutil.copy(a.voice, out / "tiny-voca-it-voice.pt"); shutil.copy(a.vocoder, out / "tiny-voca-it-vocoder.pt")
    match_frontend(a.voice); model = load_model(a.voice).eval(); vocoder = load_vocoder(a.vocoder)
    onnx = export(model, out / "tiny-voca-it.onnx", vocoder=vocoder)
    meta = metadata(model, a.voice, a.vocoder); meta["product_voices"] = names
    (out / "tiny-voca-it.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    # Samples (16-bit WAV, -20 LUFS-ish RMS), one per voice and sentence, plus emotions.
    streamer = Streamer(a.voice, a.vocoder)
    samples = out / "samples"; samples.mkdir(exist_ok=True)
    def write(name, chunks):
        audio = np.concatenate(list(chunks)); audio *= min(0.1 / max(np.sqrt(np.mean(audio ** 2)), 1e-6), 0.95 / max(np.abs(audio).max(), 1e-6))
        sf.write(samples / name, audio, streamer.sample_rate, subtype="PCM_16")
    for vid, label in names.items():
        for i, text in enumerate(SAMPLES):
            write(f"{label}-{i + 1}.wav", streamer.stream(text, voice=vid))
        for emotion, text in EMOTION_SAMPLES.items():
            if emotion in model.config.emotions:
                write(f"{label}-{emotion}.wav", streamer.stream(text, voice=vid, emotion=emotion, intensity=1.0))
    parameters = sum(x.numel() for x in model.parameters()) + sum(x.numel() for x in vocoder.parameters())
    bench = json.loads(a.benchmark.read_text()) if a.benchmark and a.benchmark.exists() else None
    card = f"""# tiny-voca-it {a.version}

A {parameters / 1e6:.2f}M-parameter Italian text-to-speech model (voice model {sum(x.numel() for x in model.parameters()) / 1e6:.2f}M +
harmonic-prior neural vocoder {sum(x.numel() for x in vocoder.parameters()) / 1e6:.2f}M), 24 kHz, with an emotion control.
Code: https://github.com/carbogninalberto/tiny-voca-it · Browser demo: https://carbogninalberto.github.io/tiny-voca-it/

## Voices
{chr(10).join(f"- **{label}** (`{vid}`)" for vid, label in names.items())}

Both are synthetic voices (designed and generated with Qwen3-TTS, screened with Whisper); label generated
audio as synthetic. Emotions: {", ".join(model.config.emotions)} (intensity 0-1.5).

## Files
| file | what |
|---|---|
| `tiny-voca-it-voice.pt`, `tiny-voca-it-vocoder.pt` | PyTorch checkpoints (`python -m tiny_voca.infer --checkpoint ... --vocoder ...`) |
| `tiny-voca-it.onnx` + `tiny-voca-it.json` | one ONNX graph (voice model + vocoder) and its front-end metadata; runs in ONNX Runtime and ONNX Runtime Web |
| `samples/` | example audio per voice (and emotion) |
| `SHA256SUMS` | checksums |

## Use
```bash
pip install -e ".[prepare]"     # from the repository; needs eSpeak NG 1.51+ with the Italian voice
python -m tiny_voca.infer --checkpoint tiny-voca-it-voice.pt --vocoder tiny-voca-it-vocoder.pt \\
  --voice {next(iter(names))} --text "Buongiorno! Come stai?" --output ciao.wav
python -m tiny_voca.stream --checkpoint tiny-voca-it-voice.pt --vocoder tiny-voca-it-vocoder.pt \\
  --voice {next(iter(names))} --text "Prima frase. Seconda frase." | aplay -r 24000 -f S16_LE -c 1
```

## Evaluation
""" + (f"""Italian split of the MiniMax TTS Multilingual Test Set (100 sentences), Whisper-large-v3: **WER {bench['wer']:.2f}%**,
CER {bench['cer']:.2f}%. Fixed voices, not zero-shot cloning: not directly comparable with cloning systems; intelligibility
only (no naturalness score). Details: docs/benchmarks/minimax-it in the repository.
""" if bench else "See docs/benchmarks in the repository.\n") + """
## Limits
Italian only; long and short l/n/m/r are not always distinguished; a slight metallic trace on some sentences;
foreign names may be misread; no voice cloning (planned: tiny-voca-it-zs).

## Licence and credits
Code and weights Apache-2.0, based on tiny-voca by Muhammed Nazeem. Training data: CML-TTS Italian (CC BY 4.0,
LibriVox) and synthetic speech from Qwen3-TTS (Apache-2.0 weights). Vocoder teacher: Vocos (MIT).
Phonemizer: eSpeak NG (GPL-3.0), used as a separate program / unmodified WebAssembly build.
"""
    (out / "MODEL_CARD.md").write_text(card)
    files = sorted(f for f in out.rglob("*") if f.is_file() and f.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{sha256(f)}  {f.relative_to(out)}\n" for f in files))
    print(json.dumps({"release": str(out), "files": len(files) + 1, "parameters": parameters, "onnx_mb": round(onnx.stat().st_size / 1e6, 1)}))
    if a.draft:
        assets = [str(f) for f in files if f.parent == out] + [str(out / "SHA256SUMS")]
        subprocess.run(["tar", "czf", str(out / "samples.tar.gz"), "-C", str(out), "samples"], check=True)
        subprocess.run(["gh", "release", "create", f"v{a.version}", "--draft", "--title", f"tiny-voca-it {a.version}",
                        "--notes-file", str(out / "MODEL_CARD.md"), *assets, str(out / "samples.tar.gz")], check=True)


if __name__ == "__main__":
    main()
