"""Smaller variants of the one-file ONNX model.

    python scripts/quantize_onnx.py --input tiny-voca-it.onnx --output-dir out/   # writes -fp16w, -int8, -int4

- ``fp16w``: every large float weight stored in fp16 and cast back to fp32 when the graph runs (half the
  download, unchanged arithmetic; the browser's WebAssembly backend computes in fp32 anyway).
- ``int8``: dynamic quantization (int8 weights, per-batch activation scales) of MatMul and Conv.
- ``int4``: MatMul weights in 4-bit blocks of 32 (``MatMulNBits``); the remaining weights as in ``fp16w``.

The STFT kernels are computed inside the graph from the windows (``export._runtime_kernels``), so they are
not initializers and are never quantized; neither are small tensors (embeddings, norms, filter banks).
Quality is checked separately (``scripts/compare_onnx.py``).
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

MIN_ELEMENTS = 4096


def fp16_weights(model):
    graph = model.graph
    users = {}
    for node in graph.node:
        for i, name in enumerate(node.input):
            users.setdefault(name, []).append(node)
    keep, casts = [], []
    for init in graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT or int(np.prod(init.dims)) < MIN_ELEMENTS:
            keep.append(init); continue
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "_fp16")
        keep.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT, name=init.name + "_to_fp32"))
    del graph.initializer[:]
    graph.initializer.extend(keep)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--block", type=int, default=32)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    stem = a.input.stem
    out = {k: a.output_dir / f"{stem}-{k}.onnx" for k in ("fp16w", "int8", "int4")}

    onnx.save(fp16_weights(onnx.load(a.input)), out["fp16w"])

    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(a.input), str(out["int8"]), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Conv"],
                     extra_options={"MatMulConstBOnly": True})

    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
    model = onnx.load(a.input)
    quantizer = MatMulNBitsQuantizer(model, algo_config=DefaultWeightOnlyQuantConfig(block_size=a.block, is_symmetric=True, bits=4))
    quantizer.process()
    onnx.save(fp16_weights(quantizer.model.model), out["int4"])
    for k, path in out.items():
        print(f"{k}: {path} {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
