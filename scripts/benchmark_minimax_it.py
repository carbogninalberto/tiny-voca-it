"""Italian split of the MiniMax TTS Multilingual Test Set (100 sentences): synthesise every line with
tiny-voca-it (italian_female -> EF3, italian_male -> IT1), then score with Whisper-large-v3.

    python scripts/benchmark_minimax_it.py synth --text italian.txt --out bench/ --checkpoint ... --vocoder ...
    python scripts/benchmark_minimax_it.py score --text italian.txt --out bench/

Lines with several sentences are read one sentence at a time (as the playground does) and joined
with 0.25 s of silence. WER/CER use the seed-tts-eval style normalisation: lower case, punctuation
removed, whitespace collapsed. SIM (voice-cloning similarity) does not apply: the model does not clone.
"""
import argparse, json, re, time, unicodedata
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
VOICES = {"italian_female": "EF4", "italian_male": "EM4"}   # Chiara, Lorenzo (round 8); IT1/EF3 before


def lines(path):
    return [tuple(l.split("|", 1)) for l in Path(path).read_text(encoding="utf-8").splitlines() if "|" in l]


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?;])\s+", text.strip()) if s]


def normalise(text):
    text = unicodedata.normalize("NFC", text.lower()).replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text).replace("'", " ")
    return re.sub(r"\s+", " ", text).strip()


def synth(args):
    import torch
    from tiny_voca.infer import load_model, match_frontend, speak, vocode
    torch.set_num_threads(args.threads)
    match_frontend(args.checkpoint); model = load_model(args.checkpoint).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (speaker, text) in enumerate(lines(args.text)):
        voice = VOICES[speaker]; start = time.perf_counter(); parts = []
        for sentence in sentences(text):
            torch.manual_seed(0)
            wave, _, details = speak(model, sentence, voice)
            parts += [vocode(model, wave, details, args.vocoder, voice).numpy(), np.zeros(6000, dtype=np.float32)]
        audio = np.concatenate(parts[:-1]); seconds = time.perf_counter() - start
        sf.write(args.out / f"{i:03d}.wav", audio, 24000)
        rows.append({"id": i, "speaker": speaker, "voice": voice, "text": text, "audio_seconds": len(audio) / 24000, "compute_seconds": seconds})
    (args.out / "synth.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    total, compute = sum(r["audio_seconds"] for r in rows), sum(r["compute_seconds"] for r in rows)
    print(json.dumps({"utterances": len(rows), "audio_minutes": round(total / 60, 2), "real_time_factor": round(compute / total, 4), "threads": args.threads}))


def score(args):
    from faster_whisper import WhisperModel
    from jiwer import cer, wer
    import librosa
    asr = WhisperModel("large-v3", device=args.device, compute_type="float16" if args.device == "cuda" else "int8")
    rows = [json.loads(l) for l in (args.out / "synth.jsonl").read_text().splitlines()]
    for r in rows:
        audio = librosa.resample(sf.read(args.out / f"{r['id']:03d}.wav")[0].astype(np.float32), orig_sr=24000, target_sr=16000)
        r["heard"] = " ".join(s.text for s in asr.transcribe(audio, language="it", beam_size=5)[0]).strip()
        r["wer"], r["cer"] = wer(normalise(r["text"]), normalise(r["heard"])), cer(normalise(r["text"]), normalise(r["heard"]))
    (args.out / "scored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    pooled = lambda rs, f: f([normalise(r["text"]) for r in rs], [normalise(r["heard"]) for r in rs]) * 100
    digits = [r for r in rows if re.search(r"\d", r["text"])]; plain = [r for r in rows if r not in digits]
    result = {"wer": pooled(rows, wer), "cer": pooled(rows, cer),
              "wer_by_voice": {v: pooled([r for r in rows if r["voice"] == v], wer) for v in sorted({r["voice"] for r in rows})},
              "wer_without_digits": pooled(plain, wer), "lines_with_digits": len(digits), "wer_lines_with_digits": pooled(digits, wer)}
    (args.out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("synth", "score")); p.add_argument("--text", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", default="runs/voice-four/inference-135000.pt"); p.add_argument("--vocoder", default="runs/vocoder-four/vocoder-voicegain.pt")
    p.add_argument("--threads", type=int, default=1); p.add_argument("--device", default="cuda")
    a = p.parse_args(); synth(a) if a.mode == "synth" else score(a)
