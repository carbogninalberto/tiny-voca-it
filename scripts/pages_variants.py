"""Precision variants of the preview model for the browser page; prints the models.json entries.

    python scripts/pages_variants.py <models dir> <fp32 onnx> <label> <default voice> <voices...>

Writes <stem>-int8/-fp16/-fp32 (+ -int4 when the 4-bit quantizer is available) with a copy of the metadata
each, and lists them int8 first (the page default), every label carrying the precision and download size.
"""
import json, shutil, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize_onnx import fp16_weights  # noqa: E402


def main():
    root, source, label, default, voices = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5:]
    stem = source.stem
    variants = {}
    from onnxruntime.quantization import QuantType, quantize_dynamic
    # MatMul only: ONNX Runtime Web has no ConvInteger; the other large weights are stored as fp16.
    quantize_dynamic(str(source), str(root / f"{stem}-int8.onnx"), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                     extra_options={"MatMulConstBOnly": True})
    onnx.save(fp16_weights(onnx.load(root / f"{stem}-int8.onnx")), root / f"{stem}-int8.onnx")
    variants["int8"] = "int8 weights"
    onnx.save(fp16_weights(onnx.load(source)), root / f"{stem}-fp16.onnx"); variants["fp16"] = "fp16 weights"
    shutil.copy(source, root / f"{stem}-fp32.onnx"); variants["fp32"] = "full precision"
    try:
        from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
        quantizer = MatMulNBitsQuantizer(onnx.load(source), algo_config=DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=4))
        quantizer.process()
        onnx.save(fp16_weights(quantizer.model.model), root / f"{stem}-int4.onnx"); variants["int4"] = "4-bit weights"
    except Exception as error:      # optional (needs onnx_ir)
        print(f"int4 skipped: {error}", file=sys.stderr)
    stamp = datetime.now(ZoneInfo("Europe/Rome")).strftime("%Y-%m-%d %H:%M")
    entries = []
    for key, what in variants.items():
        name = f"{stem}-{key}"
        shutil.copy(source.with_suffix(".json"), root / f"{name}.json")
        size = (root / f"{name}.onnx").stat().st_size / 1e6
        entries.append({"id": f"preview-{key}", "file": name, "label": f"{label} · {key} ({what}, {size:.1f} MB) · updated {stamp}",
                        "voice": default, "voices": voices, "precision": key, "megabytes": round(size, 1)})
    source.unlink(); source.with_suffix(".json").unlink()
    print(json.dumps(entries, ensure_ascii=False))


if __name__ == "__main__":
    main()
