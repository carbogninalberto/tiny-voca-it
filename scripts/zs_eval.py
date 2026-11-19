"""Zero-shot pilot: does the fingerprint model speak in voices it never saw?

    python scripts/zs_eval.py --checkpoint runs/zs-pilot/inference.pt --vocoder ... --root data/zs-pilot --fingerprints ...

For each held-out voice: synthesise --sentences of its own (unseen) texts with the fingerprint of its single
reference clip, embed the output with the same WavLM speaker-verification model, and take the cosine to the
mean embedding of the voice's real clips. Baselines: real clip vs the voice's other clips (ceiling) and the
synthetic output vs the other held-out voices (chance). The same for a few training voices.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True); p.add_argument("--vocoder", required=True)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--fingerprints", type=Path, required=True)
    p.add_argument("--sentences", type=int, default=4); p.add_argument("--device", default="cpu"); p.add_argument("--save", type=Path)
    a = p.parse_args()
    from transformers import AutoFeatureExtractor, WavLMForXVector
    import librosa
    from tiny_voca.infer import load_model, match_frontend, speak, vocode
    from tiny_voca.vocoder import load_vocoder
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()

    def embed(wave, rate):
        wave = librosa.resample(np.asarray(wave, dtype=np.float32), orig_sr=rate, target_sr=16000)
        with torch.no_grad():
            return torch.nn.functional.normalize(sv(**extractor(wave, sampling_rate=16000, return_tensors="pt")).embeddings, dim=-1)[0]

    match_frontend(a.checkpoint); model = load_model(a.checkpoint).eval(); vocoder = load_vocoder(a.vocoder)
    table = torch.load(a.fingerprints, map_location="cpu", weights_only=False)
    voices = json.loads((a.root / "voices.json").read_text())
    rows = [json.loads(l) for l in (a.root / "manifest.jsonl").read_text().splitlines()]
    held = [v for v, i in voices.items() if i["held_out"] and v in table["fingerprints"]]
    seen = [v for v, i in voices.items() if not i["held_out"] and v in table["fingerprints"]][:8]
    if a.save:
        a.save.mkdir(parents=True, exist_ok=True)
    real = {}
    for v in held + seen:
        clips = [r for r in rows if r["voice_id"] == v]
        real[v] = torch.stack([embed(*sf.read(a.root / r["audio"])) for r in clips[:8]])
    results = {}
    for group, members in (("unseen", held), ("seen", seen)):
        own, other, ceiling = [], [], []
        for v in members:
            centre = torch.nn.functional.normalize(real[v].mean(0), dim=-1)
            # ceiling: each real clip against the mean of the voice's OTHER clips
            for k in range(len(real[v])):
                rest = torch.nn.functional.normalize(torch.cat((real[v][:k], real[v][k + 1:])).mean(0), dim=-1)
                ceiling.append(float(real[v][k] @ rest))
            texts = [r["text"] for r in rows if r["voice_id"] == v][-a.sentences:]
            fingerprint = table["fingerprints"][v][None].float()
            model.voices.override = model.frame_voices.override = fingerprint
            for i, text in enumerate(texts):
                torch.manual_seed(i)
                wave, _, details = speak(model, text, model.config.speakers[0])
                audio = vocode(model, wave, details, vocoder, None).numpy()
                if a.save and i == 0:
                    sf.write(a.save / f"{group}-{v}.wav", audio, 24000)
                e = embed(audio, 24000)
                own.append(float(e @ centre))
                for u in members:
                    if u != v:
                        other.append(float(e @ torch.nn.functional.normalize(real[u].mean(0), dim=-1)))
            model.voices.override = model.frame_voices.override = None
        results[group] = {"voices": len(members), "similarity_to_own_voice": round(float(np.mean(own)), 3),
                          "similarity_to_other_voices": round(float(np.mean(other)), 3), "real_clip_ceiling": round(float(np.mean(ceiling)), 3)}
        print(group, json.dumps(results[group]), flush=True)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
