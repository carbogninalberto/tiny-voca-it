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


def split_of(text, train_texts=frozenset(), validation_texts=frozenset()):
    """Sentence-level split. Sentences that are validation sentences of the other voices go to
    validation; sentences used for training other voices stay in train. So no validation or test
    sentence of one voice is a training sentence of another (check_leakage)."""
    if text in validation_texts:
        return "validation"
    if text in train_texts:
        return "train"
    bucket = int(hashlib.sha256(text.encode()).hexdigest(), 16) % 100
    return "test" if bucket < 4 else "validation" if bucket < 10 else "train"


# Quality gates for the training splits (the manifest keeps everything that passed screening):
# Whisper WER and speaking rate, so that misreadings and rushed or dragged clips are left out.
SPLIT_MAX_WER = 0.1
SPLIT_RATE = (11.0, 17.0)      # characters per second


def write_splits(root, train_manifests=("data/it-multi6-annotations/train.jsonl", "data/emotion-annotations/train.jsonl"),
                 validation_manifests=("data/it-multi6-annotations/validation.jsonl", "data/emotion-annotations/validation.jsonl")):
    read = lambda manifests: frozenset(json.loads(l)["text"] for m in manifests if Path(m).exists() for l in Path(m).read_text().splitlines())
    train_texts, validation_texts = read(train_manifests), read(validation_manifests) - read(train_manifests)
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()]
    kept = 0
    files = {split: open(root / f"{split}.jsonl", "w") for split in ("train", "validation", "test")}
    for row in rows:
        rate = len(row["text"]) / max(sf.info(root / row["audio"]).duration, 1e-3)
        if row["wer"] > SPLIT_MAX_WER or not SPLIT_RATE[0] <= rate <= SPLIT_RATE[1]:
            continue
        split = split_of(row["text"], train_texts, validation_texts)
        files[split].write(json.dumps({**row, "split": split, "text_id": row["id"]}, ensure_ascii=False) + "\n")
        kept += 1
    for f in files.values():
        f.close()
    return {"screened": len(rows), "kept_for_training": kept}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--reference", type=Path, required=True, help="Reference clip (its text is read from --reference-manifest)")
    parser.add_argument("--reference-manifest", type=Path, default=Path("data/emotion-synth/manifest.jsonl"))
    parser.add_argument("--hours", type=float, default=6.0, help="Stop when this much accepted audio exists")
    # Validation sentences of the other voices are read too; write_splits routes them to EF3's validation.
    parser.add_argument("--texts", nargs="+", default=["data/it-multi-annotations/train.jsonl", "data/it-multi6-annotations/validation.jsonl"])
    parser.add_argument("--exclude-texts", nargs="*", default=["data/it-multi-annotations/test.jsonl", "data/it-multi6-annotations/test.jsonl"])
    parser.add_argument("--max-wer", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--shard", default="0/1", help="k/n: this process takes every n-th sentence from k (run n processes in parallel)")
    parser.add_argument("--memory-fraction", type=float, default=0.0, help="Cap this process's share of GPU memory (0 = no cap)")
    parser.add_argument("--whisper-device", choices=("cpu", "cuda"), default="cuda",
                        help="Where the screening runs; either way it overlaps with the next batch in a worker thread")
    parser.add_argument("--whisper-threads", type=int, default=8)
    parser.add_argument("--split-train", nargs="*", default=["data/it-multi6-annotations/train.jsonl", "data/emotion-annotations/train.jsonl"],
                        help="Manifests whose texts are training texts of other voices (kept in this voice's train split)")
    parser.add_argument("--split-validation", nargs="*", default=["data/it-multi6-annotations/validation.jsonl", "data/emotion-annotations/validation.jsonl"],
                        help="Manifests whose texts are validation/test texts of other voices (routed to this voice's validation split)")
    args = parser.parse_args()
    (args.root / "audio").mkdir(parents=True, exist_ok=True)
    reference_text = next(json.loads(l)["text"] for l in args.reference_manifest.read_text().splitlines()
                          if json.loads(l)["audio"].endswith(args.reference.name))

    from concurrent.futures import ThreadPoolExecutor
    from faster_whisper import WhisperModel
    from jiwer import wer
    import librosa
    from qwen_tts import Qwen3TTSModel
    import qwen_fast
    qwen_fast.patch()   # lean sub-code loop: +20% throughput, same outputs
    from tiny_voca.evaluate import italian_normalizer as normalize
    # Generation is bound by one CPU core driving many small GPU kernels (the GPU is mostly idle), so the
    # Whisper screening runs in a worker thread, overlapped with the next batch. On the GPU it is fast
    # (on CPU int8 it took longer than the generation itself and stalled it).
    if args.whisper_device == "cpu":
        whisper = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=args.whisper_threads)
    else:
        whisper = WhisperModel("small", device="cuda", compute_type="float16")

    def error(text, wave, rate):
        audio = librosa.resample(wave.astype(np.float32), orig_sr=rate, target_sr=16000)
        heard = " ".join(s.text for s in whisper.transcribe(audio, language="it", beam_size=1)[0])
        reference = normalize(text)
        return wer(reference, normalize(heard)) if reference.strip() else 1.0

    manifest, rejected = args.root / "manifest.jsonl", args.root / "rejected.jsonl"
    seen = {json.loads(l)["id"]: json.loads(l) for f in (manifest, rejected) if f.exists() for l in f.read_text().splitlines()}
    state = {"accepted": sum(sf.info(args.root / r["audio"]).duration for r in (json.loads(l) for l in manifest.read_text().splitlines()))
             if manifest.exists() else 0.0}
    exclude = {json.loads(l)["text"].strip() for m in args.exclude_texts for l in Path(m).read_text().splitlines()}
    exclude.add(reference_text)
    texts = script_pool(args.texts, 10 ** 6, args.seed, exclude)
    jobs = [(f"{args.voice_id.lower()}-{hashlib.sha256(t.encode()).hexdigest()[:12]}", t) for t in texts]
    k, n = (int(x) for x in args.shard.split("/"))
    jobs = [j for j in jobs[k::n] if j[0] not in seen]
    # A batch runs until its longest clip ends, so batch sentences of similar length (sorted within
    # windows of 20 batches, keeping the overall order random).
    window = 20 * args.batch
    jobs = [j for start in range(0, len(jobs), window) for j in sorted(jobs[start:start + window], key=lambda j: len(j[1]))]
    if args.memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    base = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    prompt = base.create_voice_clone_prompt(ref_audio=str(args.reference), ref_text=reference_text)
    start, target = time.time(), args.hours * 3600
    print(json.dumps({"reference": str(args.reference), "reference_text": reference_text, "texts": len(texts),
                      "accepted_hours": round(state["accepted"] / 3600, 2), "batch": args.batch,
                      "whisper": args.whisper_device}, ensure_ascii=False), flush=True)
    retries = []          # clips that failed screening once; retried in a later batch with another seed

    def screen(batch, waves, rate):
        """Runs in the worker thread: score, write accepted clips, return clips to retry."""
        again = []
        for (clip, text, attempt), wave in zip(batch, waves):
            score = error(text, wave, rate)
            row = {"id": clip, "text": text, "voice_id": args.voice_id, "emotion": "neutral", "intensity": 0.0,
                   "wer": round(score, 3), "audio": f"audio/{clip}.wav", "origin": ORIGIN, "source_license": LICENSE}
            if score <= args.max_wer:
                sf.write(args.root / row["audio"], wave, rate)
                state["accepted"] += len(wave) / rate
                with open(manifest, "a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            elif attempt == 0:
                again.append((clip, text, 1))
            else:
                with open(rejected, "a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return again

    queue = [(clip, text, 0) for clip, text in jobs]
    pending, number = None, 0
    with ThreadPoolExecutor(1) as worker:
        while (queue or retries) and state["accepted"] < target:
            take = retries[:args.batch]; retries = retries[len(take):]
            need = args.batch - len(take)
            take += queue[:need]; queue = queue[need:]
            torch.manual_seed(args.seed + number + 1_000_003 * k)
            # Cap the length: 12 codec frames per second, Italian read at about 13 characters per second;
            # 2x margin plus 3 s. Without it one runaway clip holds the batch for up to 2048 frames (170 s).
            limit = int(12 * (2 * max(len(t) for _, t, _ in take) / 13 + 3))
            waves, rate = base.generate_voice_clone(text=[t for _, t, _ in take], language=["Italian"] * len(take),
                                                    voice_clone_prompt=prompt * len(take) if isinstance(prompt, list) else prompt,
                                                    max_new_tokens=limit)
            if pending is not None:
                retries += pending.result()
            pending = worker.submit(screen, take, waves, rate)
            number += 1
            if number % 10 == 0:
                print(json.dumps({"accepted_hours": round(state["accepted"] / 3600, 3), "minutes": round((time.time() - start) / 60, 1),
                                  "batches": number}), flush=True)
        if pending is not None:
            pending.result()
    print(json.dumps({"complete": True, "clips": write_splits(args.root, args.split_train, args.split_validation), "accepted_hours": round(state["accepted"] / 3600, 3)}), flush=True)


if __name__ == "__main__":
    main()
