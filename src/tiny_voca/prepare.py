"""Cloud-only ACOUSTIC feature preparation from local JSONL or HF-style Parquet shards.

Stores the waveform, log-mel, pitch/voicing and energy of each recording. It
does not tokenize text: phoneme tokens and durations are a separate, versioned
annotation layer (``tiny_voca.annotate``), so a front-end or aligner change
never forces this expensive step to run again.

No network access: use the HF CLI separately to obtain a pinned dataset snapshot.
"""

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from .config import Config
from .journal import append, atomic_json, file_hash, json_safe, recover

MIN_FRAMES = 32


def records(source):
    if source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as error:
                        yield {"_input_error":str(error)}
    elif source.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(source).iter_batches(batch_size=16):
            yield from batch.to_pylist()
    else:
        raise ValueError("input must be .jsonl or .parquet")


def extract(wave, config):
    import librosa
    c = config
    if wave.ndim != 1 or not np.isfinite(wave).all() or not wave.size:
        raise ValueError("expected finite mono audio")
    frames = math.ceil(len(wave)/c.hop)
    if frames < 1:
        raise ValueError("empty audio")
    if np.max(np.abs(wave)) > 1.01 or np.sqrt(np.mean(wave**2)) < 1e-5:
        raise ValueError("audio is clipped/out of range or effectively silent")
    wave = np.pad(wave, (0, frames*c.hop-len(wave))).astype(np.float32)
    spectrum = librosa.stft(wave, n_fft=c.n_fft, hop_length=c.hop, center=True, pad_mode="constant")[:, :frames]
    filters = librosa.filters.mel(sr=c.sample_rate, n_fft=c.n_fft, n_mels=c.n_mels)
    mel = np.log(np.maximum(filters @ np.abs(spectrum)**2, 1e-5)).T
    f0, voiced, _ = librosa.pyin(wave, fmin=c.f0_min, fmax=c.f0_max, sr=c.sample_rate,
                                frame_length=c.n_fft, hop_length=c.hop, center=True, pad_mode="constant")
    f0, voiced = f0[:frames], voiced[:frames]
    pitch = np.where(voiced, np.log(np.maximum(np.nan_to_num(f0, nan=200.0), 1)/200.0), 0)
    rms = librosa.feature.rms(y=wave, frame_length=c.n_fft, hop_length=c.hop, center=True, pad_mode="constant")[0, :frames]
    prosody = np.stack((pitch, voiced.astype(np.float32), np.log(np.maximum(rms, 1e-5))), axis=-1)
    return {"wave": torch.from_numpy(wave), "mel": torch.from_numpy(mel.astype(np.float32)),
            "prosody": torch.from_numpy(prosody.astype(np.float32))}


def reusable(stored, wave, c):
    """Features of an earlier preparation, if they were computed from exactly
    these samples (same padded waveform, consistent shapes), else ``None``."""
    frames = math.ceil(len(wave)/c.hop)
    padded = torch.from_numpy(np.pad(wave, (0, frames*c.hop-len(wave))).astype(np.float32))
    if stored["wave"].shape != padded.shape or not torch.equal(stored["wave"], padded):
        return None
    if stored["mel"].shape != (frames, c.n_mels) or stored["prosody"].shape != (frames, 3):
        return None
    return {"wave": stored["wave"], "mel": stored["mel"], "prosody": stored["prosody"]}


def _extract_job(audio, entry, c, target, reuse=None):
    """Read, check and featurize one recording; write its feature file.
    ``reuse`` is an earlier feature file of the same id with the same acoustic
    settings; it is used only when its stored waveform equals this audio.
    Returns ``(entry, reused)``, or the exception that rejects the row."""
    try:
        wave, sr = sf.read(io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else audio, dtype="float32")
        if sr != c.sample_rate:
            raise ValueError(f"expected {c.sample_rate} Hz; got {sr}. Resample explicitly before preparation")
        frames = math.ceil(len(wave)/c.hop)
        if not MIN_FRAMES <= frames <= c.max_frames:
            raise ValueError("audio duration outside admission limits")
        torch.set_num_threads(1)
        sample = reusable(torch.load(reuse, map_location="cpu", weights_only=True), wave, c) if reuse else None
        reused = sample is not None
        if sample is None:
            sample = extract(wave, c)
        sample.update(speaker=c.speakers.index(entry["speaker"]))
        target = Path(target)
        temporary = target.with_name(target.name+".tmp")
        with temporary.open("wb") as stream:
            torch.save(sample, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return {**entry, "frames":frames}, reused
    except Exception as error:
        return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--speaker-column", default="voice_id")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--resume", action="store_true", help="Skip durably accepted/rejected records from these same inputs")
    parser.add_argument("--workers", type=int, default=1, help="Processes for decoding and pitch tracking (pyin dominates)")
    parser.add_argument("--reuse", nargs="+", type=Path, default=[], help="Earlier feature directories with the same acoustic "
                        "settings: a clip with the same id whose stored waveform equals the new audio is copied, not recomputed")
    args = parser.parse_args()
    c = Config.from_dict(json.loads(args.config.read_text())) if args.config else Config()
    args.output.mkdir(parents=True, exist_ok=True)
    # Lock is held through completion or an exception. Different splits may run separately.
    with (args.output/(args.split+".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_split(args, c)


def prepare_split(args, c):
    manifest = args.output / (args.split+".jsonl")
    rejects = args.output / (args.split+".rejected.jsonl")
    metadata_path = args.output/(args.split+".preparation.json")
    identity = {"acoustic":c.acoustic_fingerprint,
                "sources":[{"path":str(p.resolve()), "sha256":file_hash(p)} for p in args.inputs],
                "text_column":args.text_column, "speaker_column":args.speaker_column, "speakers":list(c.speakers),
                # Admission limits affect which rows finish as rejects, not tensor contents.
                "admission":{"max_frames":c.max_frames, "min_frames":MIN_FRAMES}}
    if metadata_path.exists():
        if not args.resume:
            raise FileExistsError("preparation exists; use --resume or a new directory")
        if json.loads(metadata_path.read_text())["identity"] != identity:
            raise ValueError("resume inputs, feature schema or admission limits changed")
    elif manifest.exists() or rejects.exists():
        raise ValueError("manifest has no compatible preparation metadata; use a new directory")
    feature_schema = args.output/"features.json"
    if feature_schema.exists() and json.loads(feature_schema.read_text())["fingerprint"] != c.acoustic_fingerprint:
        raise ValueError("feature schema differs from other splits in this directory")
    atomic_json(feature_schema, {"fingerprint":c.acoustic_fingerprint, "fields":c.acoustic_config})
    atomic_json(metadata_path, {"identity":identity, "complete":False})
    feature_dir = args.output / args.split
    feature_dir.mkdir(exist_ok=True)
    entries, rejected = recover(manifest), recover(rejects)
    done = {r["record_key"] for r in entries+rejected}
    seen = {r["id"] for r in entries}
    for entry in entries:
        if not (args.output/entry["path"]).is_file():
            raise ValueError(f"committed feature file is missing: {entry['path']}")
    accepted_count, rejected_count = len(entries), len(rejected)
    workers = max(1, getattr(args, "workers", 1))
    earlier = {}
    for directory in getattr(args, "reuse", []):
        schema = json.loads((directory/"features.json").read_text())
        if schema["fingerprint"] != c.acoustic_fingerprint:
            raise ValueError(f"--reuse {directory}: other acoustic settings")
        for split in ("train", "validation", "test"):
            if (directory/(split+".jsonl")).exists():
                for row in recover(directory/(split+".jsonl")):
                    earlier.setdefault(row["id"], directory/row["path"])
    reused_count = 0
    with manifest.open("a", encoding="utf-8") as accepted_stream, rejects.open("a", encoding="utf-8") as rejected_stream:
        def finish(record_key, source, identifier, outcome):
            nonlocal accepted_count, rejected_count
            nonlocal reused_count
            if isinstance(outcome, Exception):
                seen.discard(identifier)
                append(rejected_stream, {"record_key":record_key, "id":str(identifier), "source":str(source), "reason":f"{type(outcome).__name__}: {outcome}"})
                rejected_count += 1
            else:
                # Journal write failures must abort, not masquerade as bad source rows.
                append(accepted_stream, outcome[0])
                accepted_count += 1
                reused_count += outcome[1]

        def admitted():
            """Cheap checks in order (duplicate ids are decided here, deterministically)."""
            for source_index, source in enumerate(args.inputs):
                for index, row in enumerate(records(source)):
                    record_key = f"{source_index}:{index}"
                    if record_key in done:
                        continue
                    identifier = None
                    try:
                        if not isinstance(row, dict):
                            raise ValueError("row must be an object")
                        if "_input_error" in row:
                            raise ValueError(row["_input_error"])
                        identifier = row.get("id")
                        if not isinstance(identifier, str) or not identifier:
                            raise ValueError("row must have a stable nonempty string id")
                        if identifier in seen:
                            raise ValueError("duplicate id")
                        if row.get("split", args.split) != args.split:
                            raise ValueError("row split differs from --split")
                        text, speaker = row[args.text_column], row[args.speaker_column]
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError("text must be a nonempty string")
                        if speaker not in c.speakers:
                            raise ValueError(f"unknown speaker {speaker!r}; expected {c.speakers}")
                        provenance = json_safe({k:row[k] for k in ("text_id", "origin", "source_id", "source_license", "source_url", "teacher_model", "reference_audio") if k in row})
                        audio = row["audio"]
                        if isinstance(audio, dict):
                            audio = audio["bytes"] if audio.get("bytes") is not None else str(source.parent / audio["path"])
                        else:
                            audio = str(source.parent / audio)
                        name = hashlib.sha256(identifier.encode()).hexdigest()+".pt"
                        entry = {"record_key":record_key, "id":identifier, "path":f"{args.split}/{name}", "acoustic":c.acoustic_fingerprint,
                                 "text_hash":hashlib.sha256(text.strip().encode()).hexdigest(), "text":text,
                                 "speaker":speaker, "source_file":str(source), "frames":None, "provenance":provenance}
                        json.dumps(entry, ensure_ascii=False, allow_nan=False)  # Validate before expensive DSP.
                    except Exception as error:
                        finish(record_key, source, identifier, error)
                        continue
                    seen.add(identifier)
                    yield record_key, source, identifier, (audio, entry, c, str(feature_dir/name), earlier.get(identifier))

        if workers == 1:
            for record_key, source, identifier, job in admitted():
                outcome = _extract_job(*job)
                finish(record_key, source, identifier, outcome)
        else:
            import concurrent.futures
            with concurrent.futures.ProcessPoolExecutor(workers) as pool:
                pending = {}
                for record_key, source, identifier, job in admitted():
                    pending[pool.submit(_extract_job, *job)] = (record_key, source, identifier)
                    if len(pending) >= 4 * workers:
                        finished, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in finished:
                            finish(*pending.pop(future), future.result())
                for future in concurrent.futures.as_completed(list(pending)):
                    finish(*pending.pop(future), future.result())
    atomic_json(metadata_path, {"identity":identity, "complete":True, "accepted":accepted_count, "rejected":rejected_count,
                                "reused_this_run":reused_count})
    print(json.dumps({"accepted":accepted_count, "rejected":rejected_count, "reused":reused_count, "manifest":str(manifest)}))
    if not accepted_count:
        raise SystemExit("no usable records; inspect rejected.jsonl")


if __name__ == "__main__":
    main()
