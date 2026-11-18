"""Does the emotion control come through? Per voice and emotion, compare the model's speech with the
emotional training data on the acoustic cues listeners use, relative to the same voice's neutral speech:
pitch level and range (semitones), loudness (dB) and speaking rate. Optionally a speech-emotion
classifier (superb/wav2vec2-base-superb-er: neutral/happy/angry/sad) scores both.

    python scripts/evaluate_emotion.py --checkpoint runs/voice-four/inference.pt --vocoder ... \
        --data data/emotion4-synth --voices EM4 EF4 --per-emotion 10

Model sentences are held-out texts of the data set (validation/test splits) read at intensity 1.0.
"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def cues(wave, rate, text):
    import librosa
    f0, voiced, _ = librosa.pyin(wave.astype(np.float32), fmin=60, fmax=600, sr=rate, frame_length=1024, hop_length=256)
    st = 12 * np.log2(f0[voiced] / 100) if voiced.any() else np.array([np.nan])
    frames = librosa.feature.rms(y=wave.astype(np.float32), frame_length=1024, hop_length=256)[0]
    loud = frames[frames > 0.1 * frames.max()]
    return {"pitch": float(np.median(st)), "range": float(np.percentile(st, 90) - np.percentile(st, 10)),
            "loudness": float(20 * np.log10(np.mean(loud) + 1e-9)), "rate": len(text) / (len(wave) / rate)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True); p.add_argument("--vocoder")
    p.add_argument("--data", type=Path, required=True); p.add_argument("--voices", nargs="+", required=True)
    p.add_argument("--per-emotion", type=int, default=10); p.add_argument("--classifier", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    from tiny_voca.infer import load_model, match_frontend, speak, vocode
    from tiny_voca.vocoder import load_vocoder
    match_frontend(a.checkpoint); model = load_model(a.checkpoint).eval()
    vocoder = load_vocoder(a.vocoder) if a.vocoder else None
    rows = [json.loads(l) for s in ("validation", "test", "train") if (a.data / f"{s}.jsonl").exists()
            for l in (a.data / f"{s}.jsonl").read_text().splitlines()]
    held = {r["text"] for s in ("validation", "test") if (a.data / f"{s}.jsonl").exists() for r in map(json.loads, (a.data / f"{s}.jsonl").read_text().splitlines())}
    classify = None
    if a.classifier:
        from transformers import pipeline
        classify = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=-1)
    results = {}
    for voice in a.voices:
        table = collections.defaultdict(lambda: collections.defaultdict(list))
        for emotion in model.config.emotions:
            data = [r for r in rows if r["voice_id"] == voice and r["emotion"] == emotion and (r["intensity"] == 1.0 or emotion == "neutral")]
            texts = [r["text"] for r in data if r["text"] in held][:a.per_emotion] or [r["text"] for r in data][:a.per_emotion]
            for r in data[:a.per_emotion]:
                w, rate = sf.read(a.data / r["audio"])
                for k, v in cues(w, rate, r["text"]).items():
                    table[("data", emotion)][k].append(v)
                if classify:
                    table[("data", emotion)]["ser"].append({x["label"]: x["score"] for x in classify(w.astype(np.float32), sampling_rate=rate, top_k=4)})
            for i, text in enumerate(texts):
                torch.manual_seed(i)
                wave, _, details = speak(model, text, voice, emotion=None if emotion == "neutral" else emotion, intensity=0.0 if emotion == "neutral" else 1.0)
                audio = (vocode(model, wave, details, vocoder, voice) if vocoder else wave).numpy()
                for k, v in cues(audio, 24000, text).items():
                    table[("model", emotion)][k].append(v)
                if classify:
                    table[("model", emotion)]["ser"].append({x["label"]: x["score"] for x in classify(audio.astype(np.float32), sampling_rate=24000, top_k=4)})
        summary = {}
        for source in ("data", "model"):
            base = {k: np.nanmean(v) for k, v in table[(source, "neutral")].items() if k != "ser"}
            for emotion in model.config.emotions:
                t = table[(source, emotion)]
                entry = {k: round(float(np.nanmean(v) - base[k]), 2) if k != "rate" else round(float(np.nanmean(v) / base[k]), 3)
                         for k, v in t.items() if k != "ser"}
                if "ser" in t:
                    entry["ser"] = {lab: round(float(np.mean([s.get(lab, 0) for s in t["ser"]])), 3) for lab in ("neu", "hap", "ang", "sad")}
                summary[f"{source} {emotion}"] = entry
        results[voice] = summary
        print(voice)
        print("  emotion      | data: pitch range loud rate | model: pitch range loud rate")
        for emotion in model.config.emotions:
            d, m = summary[f"data {emotion}"], summary[f"model {emotion}"]
            print(f"  {emotion:12s} | {d['pitch']:+5.1f} {d['range']:+5.1f} {d['loudness']:+5.1f} {d['rate']:.2f} | "
                  f"{m['pitch']:+5.1f} {m['range']:+5.1f} {m['loudness']:+5.1f} {m['rate']:.2f}" + (f" | SER data {d['ser']} model {m['ser']}" if 'ser' in d else ""))
    if a.output:
        a.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
