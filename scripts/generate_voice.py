"""Synthetic single-voice Italian corpus: Qwen3-TTS Base clones ONE reference clip on many sentences. Local GPU.

    PYTHONPATH=<dir with qwen_tts> python scripts/generate_voice.py --root data/ef3-synth --voice-id EF3 \
        --reference data/emotion-synth/audio/emo-EF3-neutral-0.0-3e06518d08.wav --hours 6

Every clip clones the same reference (audio + its text), so the timbre stays fixed, unlike
VoiceDesign, which re-imagines the voice on every call. Sentences come from the CML-TTS
training texts (7-22 words, no digits; validation and test texts excluded). faster-whisper
small (Italian) screens every clip; above --max-wer it is retried once with another seed,
then rejected. Resumable. Output: <root>/audio/*.wav, <root>/manifest.jsonl (accepted),
<root>/rejected.jsonl and train/validation/test.jsonl split by a text hash (90/6/4), in the
format tiny_voca.prepare reads. The audio is synthetic and must be labelled as such.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_emotions import BASE, script_pool  # noqa: E402

ORIGIN = "qwen3-tts-clone-synthetic"
LICENSE = "synthetic (Qwen3-TTS Base voice clone of a Qwen3-TTS VoiceDesign voice, Apache-2.0 weights)"


def split_of(text):
    bucket = int(hashlib.sha256(text.encode()).hexdigest(), 16) % 100
    return "test" if bucket < 4 else "validation" if bucket < 10 else "train"


def write_splits(root):
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()]
    for split in ("train", "validation", "test"):
        with open(root / f"{split}.jsonl", "w") as f:
            for row in rows:
                if split_of(row["text"]) == split:
                    f.write(json.dumps({**row, "split": split, "text_id": row["id"]}, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--reference", type=Path, required=True, help="Reference clip (its text is read from --reference-manifest)")
    parser.add_argument("--reference-manifest", type=Path, default=Path("data/emotion-synth/manifest.jsonl"))
    parser.add_argument("--hours", type=float, default=6.0, help="Stop when this much accepted audio exists")
    parser.add_argument("--texts", nargs="+", default=["data/it-multi-annotations/train.jsonl"])
    parser.add_argument("--exclude-texts", nargs="*", default=["data/it-multi-annotations/validation.jsonl", "data/it-multi-annotations/test.jsonl"])
    parser.add_argument("--max-wer", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    (args.root / "audio").mkdir(parents=True, exist_ok=True)
    reference_text = next(json.loads(l)["text"] for l in args.reference_manifest.read_text().splitlines()
                          if json.loads(l)["audio"].endswith(args.reference.name))

    from faster_whisper import WhisperModel
    from jiwer import wer
    import librosa
    from qwen_tts import Qwen3TTSModel
    from tiny_voca.evaluate import italian_normalizer as normalize
    whisper = WhisperModel("small", device="cuda", compute_type="float16")

    def error(text, wave, rate):
        audio = librosa.resample(wave.astype(np.float32), orig_sr=rate, target_sr=16000)
        heard = " ".join(s.text for s in whisper.transcribe(audio, language="it", beam_size=1)[0])
        reference = normalize(text)
        return wer(reference, normalize(heard)) if reference.strip() else 1.0

    manifest, rejected = args.root / "manifest.jsonl", args.root / "rejected.jsonl"
    seen = {json.loads(l)["id"]: json.loads(l) for f in (manifest, rejected) if f.exists() for l in f.read_text().splitlines()}
    accepted = sum(sf.info(args.root / r["audio"]).duration for r in (json.loads(l) for l in manifest.read_text().splitlines())) \
        if manifest.exists() else 0.0
    exclude = {json.loads(l)["text"].strip() for m in args.exclude_texts for l in Path(m).read_text().splitlines()}
    exclude.add(reference_text)
    texts = script_pool(args.texts, 10 ** 6, args.seed, exclude)
    jobs = [(f"{args.voice_id.lower()}-{hashlib.sha256(t.encode()).hexdigest()[:12]}", t) for t in texts]
    jobs = [j for j in jobs if j[0] not in seen]
    base = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    prompt = base.create_voice_clone_prompt(ref_audio=str(args.reference), ref_text=reference_text)
    start, target = time.time(), args.hours * 3600
    print(json.dumps({"reference": str(args.reference), "reference_text": reference_text, "texts": len(texts),
                      "accepted_hours": round(accepted / 3600, 2)}, ensure_ascii=False), flush=True)
    for i in range(0, len(jobs), args.batch):
        if accepted >= target:
            break
        batch = jobs[i:i + args.batch]
        for attempt in range(2):
            torch.manual_seed(args.seed + 100 * attempt + i)
            waves, rate = base.generate_voice_clone(text=[t for _, t in batch], language=["Italian"] * len(batch),
                                                    voice_clone_prompt=prompt * len(batch) if isinstance(prompt, list) else prompt)
            retry = []
            for (clip, text), wave in zip(batch, waves):
                score = error(text, wave, rate)
                row = {"id": clip, "text": text, "voice_id": args.voice_id, "emotion": "neutral", "intensity": 0.0,
                       "wer": round(score, 3), "audio": f"audio/{clip}.wav", "origin": ORIGIN, "source_license": LICENSE}
                if score <= args.max_wer:
                    sf.write(args.root / row["audio"], wave, rate)
                    accepted += len(wave) / rate
                    with open(manifest, "a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                elif attempt == 0:
                    retry.append((clip, text))
                else:
                    with open(rejected, "a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            batch = retry
            if not batch:
                break
        if (i // args.batch) % 25 == 0:
            write_splits(args.root)   # lets a CPU job prepare features while generation continues
            print(json.dumps({"accepted_hours": round(accepted / 3600, 3), "minutes": round((time.time() - start) / 60, 1)}), flush=True)
    print(json.dumps({"complete": True, "clips": write_splits(args.root), "accepted_hours": round(accepted / 3600, 3)}), flush=True)


if __name__ == "__main__":
    main()
