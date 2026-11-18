"""Keep only clips whose voice matches the voice's reference (WavLM speaker verification), then write
leakage-safe train/validation/test splits.

    python scripts/filter_speaker.py --root data/emotion4-synth --references data/voice-refs \
        --split-train <manifests> --split-validation <manifests>

VoiceDesign re-imagines the voice on every call, so emotional clips drift in timbre. Each clip's
x-vector (microsoft/wavlm-base-plus-sv) is compared with the voice's neutral reference; a clip is kept
when its cosine similarity is at least --min-similarity and it is not in the lowest --drop-fraction of
its voice x emotion group. Scores go to <root>/similarity.jsonl; splits use generate_voice.split_of
(sentences of other voices' validation/test sets stay out of this set's training split).
"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_voice import split_of  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--references", type=Path, default=Path("data/voice-refs"))
    p.add_argument("--min-similarity", type=float, default=0.80)
    p.add_argument("--drop-fraction", type=float, default=0.15)
    p.add_argument("--max-wer", type=float, default=0.2)
    p.add_argument("--split-train", nargs="*", default=[])
    p.add_argument("--split-validation", nargs="*", default=[])
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    from transformers import AutoFeatureExtractor, WavLMForXVector
    import librosa
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(a.device).eval()

    def embed(path):
        wave, rate = sf.read(path)
        wave = librosa.resample(wave.astype(np.float32), orig_sr=rate, target_sr=16000)
        inputs = extractor(wave, sampling_rate=16000, return_tensors="pt").to(a.device)
        with torch.no_grad():
            return torch.nn.functional.normalize(model(**inputs).embeddings, dim=-1)[0].cpu()

    rows = [json.loads(l) for l in (a.root / "manifest.jsonl").read_text().splitlines()]
    references = {v: embed(a.references / f"{v}.wav") for v in sorted({r["voice_id"] for r in rows})}
    scored = a.root / "similarity.jsonl"
    cache = {json.loads(l)["id"]: json.loads(l)["similarity"] for l in scored.read_text().splitlines()} if scored.exists() else {}
    with open(scored, "a") as f:
        for r in rows:
            if r["id"] not in cache:
                cache[r["id"]] = float(embed(a.root / r["audio"]) @ references[r["voice_id"]])
                f.write(json.dumps({"id": r["id"], "similarity": cache[r["id"]]}) + "\n")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["voice_id"], r["emotion"])].append(cache[r["id"]])
    floor = {k: float(np.quantile(v, a.drop_fraction)) for k, v in groups.items()}
    read = lambda ms: frozenset(json.loads(l)["text"] for m in ms if Path(m).exists() for l in Path(m).read_text().splitlines())
    train_texts, validation_texts = read(a.split_train), read(a.split_validation) - read(a.split_train)
    files = {s: open(a.root / f"{s}.jsonl", "w") for s in ("train", "validation", "test")}
    report = collections.Counter()
    for r in rows:
        s = cache[r["id"]]
        keep = s >= a.min_similarity and s >= floor[(r["voice_id"], r["emotion"])] and r["wer"] <= a.max_wer
        report[(r["voice_id"], "kept" if keep else "dropped")] += 1
        if keep:
            split = split_of(r["text"], train_texts, validation_texts)
            files[split].write(json.dumps({**r, "split": split, "text_id": r["id"], "similarity": round(s, 4)}, ensure_ascii=False) + "\n")
    for f in files.values():
        f.close()
    summary = {f"{v} {k}": n for (v, k), n in sorted(report.items())}
    summary["median_similarity"] = {f"{v} {e}": round(float(np.median(x)), 3) for (v, e), x in sorted(groups.items())}
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
