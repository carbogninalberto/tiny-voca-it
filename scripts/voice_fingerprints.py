"""Voice fingerprints (WavLM speaker-verification x-vectors, 512-d, unit length) for the zero-shot pilot.

    python scripts/voice_fingerprints.py --root data/zs-pilot --output data/zs-pilot/fingerprints.pt

Training voices: the mean of up to --clips clip embeddings (re-normalised). Held-out voices: the embedding
of their single reference clip only (the zero-shot case: one short sample of a new voice). Also writes
per-clip embeddings of the held-out voices' clips (for measuring similarity).
"""
import argparse, json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clips", type=int, default=12); p.add_argument("--device", default="cuda")
    a = p.parse_args()
    from transformers import AutoFeatureExtractor, WavLMForXVector
    import librosa
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(a.device).eval()

    def embed(path):
        wave, rate = sf.read(path)
        wave = librosa.resample(wave.astype(np.float32), orig_sr=rate, target_sr=16000)
        with torch.no_grad():
            x = model(**extractor(wave, sampling_rate=16000, return_tensors="pt").to(a.device)).embeddings
        return torch.nn.functional.normalize(x, dim=-1)[0].cpu()

    voices = json.loads((a.root / "voices.json").read_text())
    rows = [json.loads(l) for l in (a.root / "manifest.jsonl").read_text().splitlines()]
    fingerprints, clips = {}, {}
    for v, info in voices.items():
        ref = a.root / "refs" / f"{v}.wav"
        if not ref.exists():
            continue
        own = [r for r in rows if r["voice_id"] == v]
        if info["held_out"]:
            fingerprints[v] = embed(ref)
            clips[v] = torch.stack([embed(a.root / r["audio"]) for r in own]) if own else None
        else:
            vectors = [embed(a.root / r["audio"]) for r in own[:a.clips]] or [embed(ref)]
            fingerprints[v] = torch.nn.functional.normalize(torch.stack(vectors).mean(0), dim=-1)
    torch.save({"fingerprints": fingerprints, "held_out_clips": clips, "model": "microsoft/wavlm-base-plus-sv"}, a.output)
    print(json.dumps({"voices": len(fingerprints), "held_out": sum(1 for v in clips)}))


if __name__ == "__main__":
    main()
