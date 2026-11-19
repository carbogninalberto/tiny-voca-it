"""Real Italian speakers for the zero-shot model from VoxPopuli Italian (CC0, European Parliament).

    python scripts/import_voxpopuli.py --source data/voxpopuli-it/it --root data/zs-real --clips 25 --held-out 40

Keeps speakers with at least --min-clips gold-transcript clips of 2-15 s, takes up to --clips per speaker,
writes 24 kHz mono WAVs, manifest.jsonl, voices.json and train/validation/test.jsonl (held-out speakers
-> test; every 12th clip of the others -> validation), in the tiny_voca.prepare format. Voice ids VPxxx.
"""
import argparse, collections, io, json, random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--clips", type=int, default=25); p.add_argument("--min-clips", type=int, default=15)
    p.add_argument("--held-out", type=int, default=40); p.add_argument("--seed", type=int, default=20261119)
    a = p.parse_args()
    import librosa
    (a.root / "audio").mkdir(parents=True, exist_ok=True)
    files = sorted(a.source.glob("*.parquet"))
    meta = []
    for f in files:
        t = pq.read_table(f, columns=["audio_id", "raw_text", "normalized_text", "gender", "speaker_id", "is_gold_transcript"]).to_pydict()
        for i in range(len(t["audio_id"])):
            if t["is_gold_transcript"][i] and t["raw_text"][i] and t["speaker_id"][i] not in ("None", ""):
                meta.append((f, i, t["audio_id"][i], t["raw_text"][i].strip(), t["gender"][i], t["speaker_id"][i]))
    by = collections.defaultdict(list)
    for m in meta:
        by[m[5]].append(m)
    rng = random.Random(a.seed)
    speakers = sorted(s for s, rows in by.items() if len(rows) >= a.min_clips)
    rng.shuffle(speakers)
    held = set(speakers[:a.held_out])
    ids = {s: f"VP{k:03d}" for k, s in enumerate(sorted(speakers))}
    chosen = {}
    for s in speakers:
        rows = by[s][:]; rng.shuffle(rows); chosen[s] = rows[:a.clips * 2]      # spare clips for the length filter
    need = collections.defaultdict(list)
    for s, rows in chosen.items():
        for m in rows:
            need[m[0]].append(m)
    out, voices = [], {}
    taken = collections.Counter()
    for f, rows in need.items():
        wanted = {m[1]: m for m in rows}
        # Stream the audio column in small batches (a shard is ~3 GB; reading it whole exhausted RAM).
        offset, items = 0, {}
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["audio"]):
            column = batch.column(0)
            for j in range(len(column)):
                if offset + j in wanted:
                    items[offset + j] = column[j].as_py()
            offset += len(column)
            del batch, column
            if len(items) == len(wanted):
                break
        for i, m in sorted(wanted.items()):
            s = m[5]
            if taken[s] >= a.clips or i not in items:
                continue
            item = items.pop(i)
            wave, rate = sf.read(io.BytesIO(item["bytes"]))
            if wave.ndim > 1:
                wave = wave.mean(1)
            seconds = len(wave) / rate
            if not 2.0 <= seconds <= 15.0:
                continue
            wave = librosa.resample(wave.astype(np.float32), orig_sr=rate, target_sr=24000)
            vid = ids[s]; clip = f"{vid.lower()}-{m[2]}"
            sf.write(a.root / "audio" / f"{clip}.wav", wave, 24000); taken[s] += 1
            out.append({"id": clip, "text": m[3], "voice_id": vid, "speaker": vid, "speaker_id": s, "emotion": "neutral", "intensity": 0.0,
                        "audio": f"audio/{clip}.wav", "origin": "voxpopuli-it", "source_license": "CC0 1.0 (VoxPopuli)", "held_out": s in held})
            voices[vid] = {"gender": m[4], "source_speaker": s, "held_out": s in held, "description": f"VoxPopuli Italian speaker {s} ({m[4]})"}
    (a.root / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    (a.root / "voices.json").write_text(json.dumps(voices, ensure_ascii=False, indent=1))
    files = {k: open(a.root / f"{k}.jsonl", "w") for k in ("train", "validation", "test")}
    count = collections.Counter()
    for r in out:
        k = count[r["voice_id"]]; count[r["voice_id"]] += 1
        split = "test" if r["held_out"] else ("validation" if k % 12 == 11 else "train")
        files[split].write(json.dumps({**r, "split": split, "text_id": r["id"]}, ensure_ascii=False) + "\n")
    for f in files.values():
        f.close()
    print(json.dumps({"speakers": len(voices), "held_out": sum(v["held_out"] for v in voices.values()), "clips": len(out),
                      "hours": round(sum(sf.info(a.root / r["audio"]).duration for r in out) / 3600, 2)}))


if __name__ == "__main__":
    main()
