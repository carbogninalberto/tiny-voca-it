"""Import one speaker of CML-TTS Italian (HF Parquet layout) as prepare-ready JSONL.

    python scripts/import_italian_dataset.py --source data/cml-tts/italian --speaker 1595 \
        --voice IT1 --output data/it-1595 --max-hours 12

Source: CML-TTS (Oliveira et al. 2023), Italian, CC BY 4.0, 24 kHz audiobook
speech from LibriVox recordings, mirrored as Parquet at
https://huggingface.co/datasets/ylacombe/cml-tts (rows: audio{bytes,path}, text,
transcript_wav2vec, levenshtein, duration, speaker_id). The audio path is
``<speaker>_<book>_<index>.wav``; a run of ``--section-size`` consecutive
clips of one book is a split group (an approximate chapter).

Writes ``<output>/audio/<id>.wav`` (16-bit, mono, 24 kHz), ``<split>.jsonl``
with ``id/text/voice_id/audio/split`` and provenance, ``rejected.jsonl`` with
reasons, and ``receipt.json`` (source revision, per-shard checksums, filters,
seed, decoder, split report and the locked test-manifest digest). Source files
are never modified. Re-running with the same settings is a no-op for audio that
already exists; changed settings require a new output directory.
"""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_voca.dataset_import import (assign_splits, audio_hash, check_audio, connected_groups,  # noqa: E402
                                      keep_held_out, split_report)
from tiny_voca.journal import atomic_json, file_hash  # noqa: E402

COLUMNS = ("text", "speaker_id", "duration", "levenshtein")
SOURCE_URL = "https://huggingface.co/datasets/ylacombe/cml-tts"
LICENSE = "CC-BY-4.0"
TARGET_RATE = 24000


def shard_checksum(shard):
    """sha256 recorded by the HF download (``.cache/.../<name>.metadata`` line 2), else computed."""
    metadata = shard.parent.parent / ".cache" / "huggingface" / "download" / shard.parent.name / (shard.name + ".metadata")
    if metadata.exists():
        lines = metadata.read_text().split("\n")
        return {"revision": lines[0], "sha256": lines[1], "how": "hf-download-metadata"}
    return {"revision": None, "sha256": file_hash(shard), "how": "computed"}


def safe_name(path_field):
    name = Path(path_field).name
    if name != path_field or not name.endswith(".wav") or name.startswith("."):
        raise ValueError(f"unsafe or unexpected audio path {path_field!r}")
    parts = name[:-4].split("_")
    # <speaker>_<book>_<index>.wav, index optionally with a segment suffix ("000115-0003").
    if len(parts) != 3 or not (parts[0].isdigit() and parts[1].isdigit() and parts[2].replace("-", "").isdigit()):
        raise ValueError(f"audio name is not <speaker>_<book>_<index>.wav: {path_field!r}")
    return name[:-4], parts


def decode(payload):
    wave, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)                       # explicit mono
    if rate != TARGET_RATE:
        import soxr
        wave = soxr.resample(wave, rate, TARGET_RATE, quality="VHQ").astype(np.float32)
    return wave, rate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Directory with the italian/*.parquet shards")
    parser.add_argument("--speaker", type=int, required=True)
    parser.add_argument("--voice", required=True, help="voice_id written to the manifests (model voice name)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-levenshtein", type=float, default=0.93,
                        help="Minimum similarity between text and the source's wav2vec transcript (drops misreadings)")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-hours", type=float, help="Cap usable audio; deterministic seeded choice among passing clips")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--section-size", type=int, default=100,
                        help="Consecutive clip numbers per split group inside a book (approximates a chapter/session)")
    parser.add_argument("--keep-splits", type=Path,
                        help="Earlier import of the same speaker: its validation and test clips stay exactly as they were "
                             "(the test manifest must come out byte-identical), new clips of their groups are dropped, "
                             "everything else is training data")
    args = parser.parse_args()
    import pyarrow.parquet as pq

    shards = sorted(args.source.glob("*.parquet"))
    if not shards:
        raise SystemExit("no parquet shards found")
    settings = {k: getattr(args, k) for k in ("speaker", "voice", "min_levenshtein", "max_seconds", "min_seconds", "max_hours", "seed",
                                              "section_size")}
    kept = None
    if args.keep_splits:
        earlier = json.loads((args.keep_splits / "receipt.json").read_text())
        if {k: v for k, v in earlier["settings"].items() if k not in ("max_hours", "keep_splits")} != \
                {k: v for k, v in settings.items() if k != "max_hours"}:
            raise SystemExit("--keep-splits: the earlier import used other filters, voice, seed or grouping")
        kept = {}
        for split in ("validation", "test"):
            for line in (args.keep_splits / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    kept[json.loads(line)["id"]] = split
        settings["keep_splits"] = {"source": str(args.keep_splits), "test_manifest_sha256": earlier["test_manifest_sha256"],
                                   "validation_manifest_sha256": file_hash(args.keep_splits / "validation.jsonl")}
    receipt_path = args.output / "receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text())["settings"] != settings:
        raise SystemExit("output exists with different settings; use a new directory")
    (args.output / "audio").mkdir(parents=True, exist_ok=True)

    # 1. Metadata pass (no audio read): candidates and filter reasons.
    candidates, rejected = [], []
    for shard in shards:
        table = pq.read_table(shard, columns=list(COLUMNS))   # column pruning: no audio is read here
        for index, row in enumerate(table.to_pylist()):
            if row["speaker_id"] != args.speaker:
                continue
            reasons = []
            if not isinstance(row["text"], str) or not row["text"].strip():
                reasons.append("empty_text")
            if row["levenshtein"] is None or row["levenshtein"] < args.min_levenshtein:
                reasons.append("transcript_mismatch")
            if not args.min_seconds <= (row["duration"] or 0) <= args.max_seconds:
                reasons.append("duration")
            entry = {"shard": shard.name, "row": index, "text": row["text"], "seconds": row["duration"], "levenshtein": row["levenshtein"]}
            (rejected.append({**entry, "reasons": reasons}) if reasons else candidates.append(entry))
    if args.max_hours:
        candidates.sort(key=lambda c: hashlib.sha256(f"{args.seed}:{c['shard']}:{c['row']}".encode()).hexdigest())
        kept, total = [], 0.0
        for c in candidates:
            if total >= args.max_hours * 3600:
                rejected.append({**c, "reasons": ["over_hour_cap"]})
                continue
            kept.append(c)
            total += c["seconds"]
        candidates = kept
    wanted = {}
    for c in candidates:
        wanted.setdefault(c["shard"], {})[c["row"]] = c
    print(json.dumps({"candidates": len(candidates), "filtered": len(rejected),
                      "candidate_hours": round(sum(c["seconds"] for c in candidates) / 3600, 2)}), flush=True)

    # 2. Audio pass: decode, validate, write 16-bit WAV.
    records, seen = [], set()
    for shard in shards:
        rows = wanted.get(shard.name)
        if not rows:
            continue
        audio = pq.read_table(shard, columns=["audio"]).column("audio")
        for index, c in rows.items():
            item = audio[index].as_py()
            try:
                identifier, (speaker, book, number) = safe_name(item["path"])
                if int(speaker) != args.speaker:
                    raise ValueError("audio name speaker differs from speaker_id")
                if identifier in seen:
                    raise ValueError("duplicate id")
                wave, rate = decode(item["bytes"])
                check_audio(wave, TARGET_RATE)
                target = args.output / "audio" / f"{identifier}.wav"
                if not target.exists():
                    temporary = target.with_suffix(".tmp.wav")
                    sf.write(temporary, wave, TARGET_RATE, subtype="PCM_16")
                    temporary.replace(target)
                seen.add(identifier)
                records.append({"id": f"cml-it-{identifier}", "file": target.name, "text": c["text"].strip(),
                                "seconds": len(wave) / TARGET_RATE, "group": f"book-{book}-section-{int(number.split('-')[0]) // args.section_size}", "audio_sha256": audio_hash(wave),
                                "source_rate": rate, "levenshtein": c["levenshtein"], "source": f"{shard.name}#{index}"})
            except Exception as error:   # recorded with reason, never silently dropped
                rejected.append({**c, "reasons": [f"{type(error).__name__}: {error}"]})
        print(json.dumps({"shard": shard.name, "accepted_so_far": len(records)}), flush=True)

    # 3. Duplicate groups and deterministic splits.
    groups = connected_groups(records)
    if kept is None:
        splits = assign_splits(records, groups, args.seed)
    else:
        splits = keep_held_out(records, groups, kept)
        dropped = [r for r in records if splits[r["id"]] is None]
        rejected.extend({"id": r["id"], "source": r["source"], "seconds": r["seconds"], "reasons": ["held_out_group_extension"]}
                        for r in dropped)
        records = [r for r in records if splits[r["id"]] is not None]
    leaked = Counter()
    for split in ("train", "validation", "test"):
        rows = sorted((r for r in records if splits[r["id"]] == split), key=lambda r: r["id"])
        for r in rows:
            leaked[(groups[r["id"]], split)] += 1
        payload = "".join(json.dumps({"id": r["id"], "text": r["text"], "voice_id": args.voice, "audio": f"audio/{r['file']}",
                                      "split": split, "text_id": groups[r["id"]], "origin": "cml-tts-italian",
                                      "source_id": r["source"], "source_license": LICENSE, "source_url": SOURCE_URL},
                                     ensure_ascii=False) + "\n" for r in rows)
        if kept is not None and split != "train":
            # Same clips as before, written verbatim: a group key (text_id) may have been
            # renamed by new clips that joined the group and were then dropped.
            earlier_text = (args.keep_splits / f"{split}.jsonl").read_text(encoding="utf-8")
            if {json.loads(line)["id"] for line in earlier_text.splitlines() if line.strip()} != {r["id"] for r in rows}:
                raise RuntimeError(f"the {split} clips differ from the earlier import")
            payload = earlier_text
        (args.output / f"{split}.jsonl").write_text(payload, encoding="utf-8")
    per_group = Counter(group for group, _ in leaked)
    if any(count > 1 for count in per_group.values()):
        raise RuntimeError("a duplicate group spans several splits")
    (args.output / "rejected.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected), encoding="utf-8")
    report = split_report(records, splits)
    receipt = {
        "source": {"url": SOURCE_URL, "license": LICENSE, "dataset": "CML-TTS Italian (LibriVox audiobooks)",
                   "attribution": "Oliveira, F. S. et al., CML-TTS: A Multilingual Dataset for Speech Synthesis in "
                                  "Low-Resource Languages, 2023",
                   "shards": {s.name: shard_checksum(s) for s in shards}},
        "settings": settings, "decoder": {"soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
                                          "mono": "channel mean", "resample": f"soxr VHQ only if not {TARGET_RATE} Hz",
                                          "written": "PCM_16"},
        "source_rates": dict(Counter(r["source_rate"] for r in records)),
        "groups": {"count": len(set(groups.values())), "rule": "book section (clip number // section_size) + identical normalised transcript + identical audio"},
        "splits": report, "rejections": dict(Counter(reason.split(":")[0] for r in rejected for reason in r["reasons"])),
        "test_manifest_sha256": file_hash(args.output / "test.jsonl"),
    }
    if kept is not None:
        for split in ("validation", "test"):
            if file_hash(args.output / f"{split}.jsonl") != settings["keep_splits"][f"{split}_manifest_sha256"]:
                raise RuntimeError(f"the {split} manifest changed although --keep-splits was given")
    atomic_json(receipt_path, receipt)
    print(json.dumps({"splits": report, "rejections": receipt["rejections"], "groups": receipt["groups"]["count"]}, indent=1))


if __name__ == "__main__":
    main()
