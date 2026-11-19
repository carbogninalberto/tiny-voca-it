"""Add the zero-shot preview (zs.html) to a built Pages directory.

    python scripts/build_zs_preview.py --pages build/pages --checkpoint runs/zs-pilot/inference.pt \
        --vocoder runs/vocoder-four/vocoder-voicegain.pt --pool data/zs-pilot --fingerprints data/zs-pilot/fingerprints.pt \
        --fingerprint-onnx fingerprint-int4.onnx

Writes models/zs-pilot.onnx (+ .json; lean graph, int8 MatMul + fp16 storage), models/zs-fingerprint.onnx, zs/voices.json
(held-out voices first, each with its reference-clip fingerprint) and zs/refs/*.wav (16 kHz), and appends hidden
entries for both models to models/models.json (the main page lists only entries without "hidden").
"""
import argparse, hashlib, json, re, shutil, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def short(description):
    m = re.match(r"A (man|woman) in (?:his|her) (\w+) with (?:a |an )?([^,]+?) voice, native Italian speaker from (\w+)", description)
    return f"{m[2]}, {m[3].replace('slightly ', '')}, {m[4]}" if m else description[:40]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pages", type=Path, required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--vocoder", required=True)
    p.add_argument("--pool", type=Path, required=True); p.add_argument("--fingerprints", type=Path, required=True)
    p.add_argument("--fingerprint-onnx", type=Path, required=True); p.add_argument("--seen", type=int, default=10)
    a = p.parse_args()
    import onnx
    import librosa
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from quantize_onnx import fp16_weights
    from tiny_voca.export import export, metadata
    from tiny_voca.infer import load_model, match_frontend
    from tiny_voca.vocoder import load_vocoder
    models = a.pages / "models"
    match_frontend(a.checkpoint); model = load_model(a.checkpoint).eval(); vocoder = load_vocoder(a.vocoder)
    full = models / "zs-pilot-fp32.onnx"
    export(model, full, vocoder=vocoder)
    quantize_dynamic(str(full), str(models / "zs-pilot.onnx"), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"], extra_options={"MatMulConstBOnly": True})
    onnx.save(fp16_weights(onnx.load(models / "zs-pilot.onnx")), models / "zs-pilot.onnx"); full.unlink()
    (models / "zs-pilot.json").write_text(json.dumps(metadata(model, a.checkpoint, a.vocoder), ensure_ascii=False, indent=1))
    shutil.copy(a.fingerprint_onnx, models / "zs-fingerprint.onnx")
    table = torch.load(a.fingerprints, map_location="cpu", weights_only=False)["fingerprints"]
    info = json.loads((a.pool / "voices.json").read_text())
    held = [v for v in info if info[v]["held_out"] and v in table]
    seen = [v for v in info if not info[v]["held_out"] and v in table][:a.seen]
    (a.pages / "zs" / "refs").mkdir(parents=True, exist_ok=True)
    voices = []
    for v in held + seen:
        wave, rate = sf.read(a.pool / "refs" / f"{v}.wav")
        sf.write(a.pages / "zs" / "refs" / f"{v}.wav", librosa.resample(wave.astype(np.float32), orig_sr=rate, target_sr=16000), 16000, subtype="PCM_16")
        voices.append({"id": v, "gender": info[v]["gender"], "short": short(info[v]["description"]), "description": info[v]["description"],
                       "held_out": info[v]["held_out"], "ref": f"zs/refs/{v}.wav", "fingerprint": [round(float(x), 5) for x in table[v]]})
    (a.pages / "zs" / "voices.json").write_text(json.dumps(voices, ensure_ascii=False))
    entries = json.loads((models / "models.json").read_text())
    entries = [e for e in entries if not e["id"].startswith("zs-")]
    def digest(*files):
        h = hashlib.sha256()
        for f in files:
            h.update(f.read_bytes())
        return h.hexdigest()[:16]
    entries.append({"id": "zs-pilot", "file": "zs-pilot", "label": "tiny-voca-it-zs pilot", "hidden": True,
                    "hash": digest(models / "zs-pilot.onnx", models / "zs-pilot.json")})
    entries.append({"id": "zs-fingerprint", "file": "zs-fingerprint", "label": "voice fingerprint (WavLM-SV, int4)", "hidden": True,
                    "hash": digest(models / "zs-fingerprint.onnx")})
    (models / "models.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"voices": len(voices), "unseen": len(held), "tts_mb": round((models / "zs-pilot.onnx").stat().st_size / 1e6, 1),
                      "fingerprint_mb": round((models / "zs-fingerprint.onnx").stat().st_size / 1e6, 1)}))


if __name__ == "__main__":
    main()
