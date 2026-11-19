"""Many distinct synthetic Italian voices for the zero-shot (tiny-voca-it-zs) pilot. Local GPU.

    PYTHONPATH=<qwen_tts> python scripts/generate_voice_pool.py --root data/zs-pilot --voices 200 --clips 24

1. Each voice gets a description drawn from attributes (gender, age, pitch and timbre, region, style);
   Qwen3-TTS VoiceDesign speaks one reference sentence with it (screened with Whisper, retried).
2. Qwen3-TTS Base clones every reference on --clips CML-TTS sentences, so each voice keeps ONE timbre.
3. faster-whisper screens every clip (WER <= --max-wer, one retry).

Voices ZS000..; the last --held-out voices are never used for training (they test cloning of unseen
voices). Resumable. Output: <root>/refs/, <root>/audio/, <root>/voices.json, <root>/manifest.jsonl,
and train/validation/test.jsonl in the tiny_voca.prepare format. Synthetic audio; label it as such.
"""
import argparse, hashlib, json, random, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_emotions import BASE, DESIGN, REFERENCE_TEXT, script_pool  # noqa: E402

GENDERS = [("man", "male"), ("woman", "female")]
AGES = ["in his twenties", "in his thirties", "in his forties", "in his fifties", "in his sixties", "in his seventies"]
TIMBRES = ["a deep, resonant voice", "a warm, medium-low voice", "a clear, medium-pitched voice", "a bright, light voice",
           "a soft, breathy voice", "a slightly husky voice", "a crisp, precise voice", "a rich, velvety voice", "a thin, nasal voice",
           "a gentle, calm voice", "a strong, projected voice", "a slightly raspy voice"]
REGIONS = ["Milan", "Turin", "Venice", "Bologna", "Florence", "Rome", "Naples", "Bari", "Palermo", "Cagliari", "Genoa", "Trieste",
           "Perugia", "Verona", "Ancona", "Catania"]
STYLES = ["calm and measured delivery", "lively, energetic delivery", "warm, friendly delivery", "neutral newsreader delivery",
          "relaxed, conversational delivery", "slow, thoughtful delivery"]


def describe(rng):
    who, gender = rng.choice(GENDERS)
    age = rng.choice(AGES).replace("his", "his" if who == "man" else "her")
    text = (f"A {who} {age} with {rng.choice(TIMBRES)}, native Italian speaker from {rng.choice(REGIONS)}, {rng.choice(STYLES)}, "
            "clear diction, close microphone, quiet room.")
    return gender, text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--voices", type=int, default=200)
    p.add_argument("--clips", type=int, default=24); p.add_argument("--held-out", type=int, default=20)
    p.add_argument("--texts", nargs="+", default=["data/it-multi-annotations/train.jsonl"])
    p.add_argument("--exclude-texts", nargs="*", default=["data/it-multi-annotations/test.jsonl", "data/it-multi6-annotations/test.jsonl"])
    p.add_argument("--max-wer", type=float, default=0.2); p.add_argument("--batch", type=int, default=16)
    p.add_argument("--memory-fraction", type=float, default=0.8); p.add_argument("--seed", type=int, default=20261119)
    a = p.parse_args()
    for sub in ("refs", "audio"):
        (a.root / sub).mkdir(parents=True, exist_ok=True)
    from faster_whisper import WhisperModel
    from jiwer import wer
    import librosa
    from qwen_tts import Qwen3TTSModel
    import qwen_fast
    qwen_fast.patch_graphs()
    from tiny_voca.evaluate import italian_normalizer as normalize
    if a.memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    whisper = WhisperModel("small", device="cuda", compute_type="float16")

    def error(text, wave, rate):
        heard = " ".join(s.text for s in whisper.transcribe(librosa.resample(wave.astype(np.float32), orig_sr=rate, target_sr=16000), language="it", beam_size=1)[0])
        return wer(normalize(text), normalize(heard)) if normalize(text).strip() else 1.0

    # 1. voice designs and their reference clips
    rng = random.Random(a.seed)
    voices_file = a.root / "voices.json"
    voices = json.loads(voices_file.read_text()) if voices_file.exists() else {}
    for i in range(a.voices):
        vid = f"ZS{i:03d}"
        if vid not in voices:
            gender, text = describe(rng)
            voices[vid] = {"gender": gender, "description": text, "held_out": i >= a.voices - a.held_out}
        else:
            describe(rng)        # keep the random stream aligned on resume
    voices_file.write_text(json.dumps(voices, ensure_ascii=False, indent=1))
    pending = [v for v in voices if not (a.root / "refs" / f"{v}.wav").exists()]
    if pending:
        design = Qwen3TTSModel.from_pretrained(DESIGN, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        for attempt in range(3):
            todo = [v for v in pending if not (a.root / "refs" / f"{v}.wav").exists()]
            for start in range(0, len(todo), a.batch):
                batch = todo[start:start + a.batch]
                torch.manual_seed(a.seed + attempt * 1000 + start)
                waves, rate = design.generate_voice_design(text=[REFERENCE_TEXT] * len(batch), instruct=[voices[v]["description"] for v in batch],
                                                           language=["Italian"] * len(batch), max_new_tokens=400)
                for v, w in zip(batch, waves):
                    if error(REFERENCE_TEXT, w, rate) <= 0.1:
                        sf.write(a.root / "refs" / f"{v}.wav", w, rate)
            print(json.dumps({"references": sum((a.root / "refs" / f"{v}.wav").exists() for v in voices), "attempt": attempt}), flush=True)
        del design; torch.cuda.empty_cache()

    # 2. clones on ordinary sentences
    ready = [v for v in voices if (a.root / "refs" / f"{v}.wav").exists()]
    exclude = {json.loads(l)["text"].strip() for m in a.exclude_texts for l in Path(m).read_text().splitlines()} | {REFERENCE_TEXT}
    pool = script_pool(a.texts, 10 ** 6, a.seed, exclude)
    manifest, rejected = a.root / "manifest.jsonl", a.root / "rejected.jsonl"
    done = {json.loads(l)["id"] for f in (manifest, rejected) if f.exists() for l in f.read_text().splitlines()}
    jobs = []
    for n, v in enumerate(ready):
        for k in range(a.clips):
            text = pool[(n * a.clips + k) % len(pool)]
            clip = f"{v.lower()}-{hashlib.sha256(text.encode()).hexdigest()[:10]}"
            if clip not in done:
                jobs.append((v, clip, text, 0))
    jobs.sort(key=lambda j: len(j[2]))
    base = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    prompts = {v: base.create_voice_clone_prompt(ref_audio=str(a.root / "refs" / f"{v}.wav"), ref_text=REFERENCE_TEXT) for v in ready}
    start, made = time.time(), 0
    while jobs:
        batch, jobs = jobs[:a.batch], jobs[a.batch:]
        limit = int(12 * (2 * max(len(t) for _, _, t, _ in batch) / 13 + 3))
        prompt = [item for v, _, _, _ in batch for item in prompts[v]]
        waves, rate = base.generate_voice_clone(text=[t for _, _, t, _ in batch], language=["Italian"] * len(batch), voice_clone_prompt=prompt, max_new_tokens=limit)
        for (v, clip, text, attempt), w in zip(batch, waves):
            score = error(text, w, rate)
            row = {"id": clip, "text": text, "voice_id": v, "speaker": v, "emotion": "neutral", "intensity": 0.0, "wer": round(score, 3),
                   "audio": f"audio/{clip}.wav", "origin": "qwen3-tts-clone-synthetic", "held_out": voices[v]["held_out"]}
            if score <= a.max_wer:
                sf.write(a.root / row["audio"], w, rate); made += 1
                with open(manifest, "a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            elif attempt == 0:
                jobs.append((v, clip, text, 1))
            else:
                with open(rejected, "a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if made and made % 200 < a.batch:
            print(json.dumps({"clips": made, "left": len(jobs), "minutes": round((time.time() - start) / 60, 1)}), flush=True)

    # 3. splits: held-out voices -> test; per voice, every 12th clip -> validation
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    files = {s: open(a.root / f"{s}.jsonl", "w") for s in ("train", "validation", "test")}
    seen = {}
    for r in rows:
        k = seen[r["voice_id"]] = seen.get(r["voice_id"], -1) + 1
        split = "test" if r["held_out"] else ("validation" if k % 12 == 11 else "train")
        files[split].write(json.dumps({**r, "split": split, "text_id": r["id"]}, ensure_ascii=False) + "\n")
    for f in files.values():
        f.close()
    print(json.dumps({"complete": True, "voices": len(ready), "clips": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
