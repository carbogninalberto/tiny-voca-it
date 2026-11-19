"""Quality of ONNX variants against the full-precision graph, on the MiniMax Italian sentences.

    python scripts/compare_onnx.py --reference model.onnx --variants a.onnx b.onnx --text italian.txt [--wer]

Per variant: share of sentences whose phone durations equal the reference's, the log-mel distance to the
reference on those sentences (above 1 kHz and overall) next to the reference-vs-itself floor (the graph
draws its own noise, so two runs never match exactly), and optionally Whisper WER.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True); p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--text", required=True); p.add_argument("--limit", type=int, default=40); p.add_argument("--wer", action="store_true")
    a = p.parse_args()
    from benchmark_minimax_it import lines, normalise, sentences
    from tiny_voca.italian_numbers import expand_numbers
    from tiny_voca.losses import VocosMel
    from tiny_voca.text import phonemize
    from tiny_voca.languages import profile_for
    meta = json.loads(Path(a.reference).with_suffix(".json").read_text())
    voices = {"italian_female": meta["voices"].index("EF4"), "italian_male": meta["voices"].index("EM4")}
    profile = profile_for("it")
    items = [(voices[s], sentences(expand_numbers(t))[0]) for s, t in lines(a.text)[:a.limit]]
    mel = VocosMel(); low = int((mel.filters.argmax(1).float() * 24000 / 1024 < 1000).sum())

    def run(path):
        o = ort.SessionOptions(); o.intra_op_num_threads = 4
        s = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"]); out = []
        for voice, text in items:
            ph = phonemize(text, profile)
            f = {"tokens": np.array([ph.ids], np.int64), "stress": np.array([ph.stress], np.int64), "voice": np.array([voice]),
                 "speed": np.ones(1, np.float32), "pitch_shift": np.zeros(1, np.float32), "pitch_range": np.ones(1, np.float32),
                 "emotion": np.array([0]), "intensity": np.zeros(1, np.float32)}
            audio, durations = s.run(None, f); out.append((audio[0], durations[0], text))
        return out

    def distance(x, y):
        n = min(len(x), len(y)); mx, my = mel(torch.tensor(x[:n])[None]), mel(torch.tensor(y[:n])[None])
        return float((mx - my).abs().mean()), float((mx[..., low:] - my[..., low:]).abs().mean())

    reference, again = run(a.reference), run(a.reference)
    asr = None
    if a.wer:
        from faster_whisper import WhisperModel
        from jiwer import wer
        import librosa
        asr = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=8)
        def error(outs):
            hyp = [" ".join(s.text for s in asr.transcribe(librosa.resample(x, orig_sr=24000, target_sr=16000), language="it", beam_size=5)[0]) for x, _, _ in outs]
            return 100 * wer([normalise(t) for _, _, t in outs], [normalise(h) for h in hyp])
    rows = []
    for name, outs in [("reference (2nd run)", again)] + [(Path(v).name, run(v)) for v in a.variants]:
        same = [i for i, (r, o) in enumerate(zip(reference, outs)) if np.array_equal(r[1], o[1])]
        d = [distance(reference[i][0], outs[i][0]) for i in same] or [(np.nan, np.nan)]
        row = {"variant": name, "durations_equal": f"{len(same)}/{len(items)}", "mel_l1": round(float(np.mean([x[0] for x in d])), 4),
               "mel_l1_above_1k": round(float(np.mean([x[1] for x in d])), 4)}
        if asr:
            row["whisper_small_wer"] = round(error(outs), 2)
        rows.append(row); print(json.dumps(row), flush=True)
    if asr:
        print(json.dumps({"variant": "reference", "whisper_small_wer": round(error(reference), 2)}))


if __name__ == "__main__":
    main()
