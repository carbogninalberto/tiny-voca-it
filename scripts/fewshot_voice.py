"""Few-shot voice adaptation for fingerprint models: optimise ONLY the two voice vectors (token and frame level,
2 × 192 numbers) to reproduce an aligned reference clip, with the whole model frozen.

    python scripts/fewshot_voice.py --checkpoint runs/zs-real/inference.pt --annotations <dir>/train.jsonl \
        --fingerprint <reference wav> --steps 400 --output voice.pt

Starts from the projection of the clip's fingerprint, so the result can only improve on zero-shot. Writes the
two vectors (a ~1.5 KB voice file) that ``speak`` can use via the model's voice override.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def fingerprint_of(path, device):
    import librosa
    from transformers import AutoFeatureExtractor, WavLMForXVector
    ex = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"); sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
    w, r = sf.read(path); w = librosa.resample((w.mean(1) if w.ndim > 1 else w).astype(np.float32), orig_sr=r, target_sr=16000)
    with torch.no_grad():
        return torch.nn.functional.normalize(sv(**ex(w, sampling_rate=16000, return_tensors="pt")).embeddings, dim=-1)[0].to(device)


def install(model, token_vector, frame_vector):
    """Replace both voice tables by fixed vectors (batch-broadcast)."""
    model.voices.forward = lambda speakers: token_vector.expand(len(speakers), -1)
    model.frame_voices.forward = lambda speakers: frame_vector.expand(len(speakers), -1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True); p.add_argument("--annotations", nargs="+", required=True)
    p.add_argument("--fingerprint", required=True, help="Reference clip for the starting fingerprint")
    p.add_argument("--steps", type=int, default=400); p.add_argument("--learning-rate", type=float, default=3e-2)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--device", default="cuda")
    a = p.parse_args()
    from tiny_voca.data import AnnotatedDataset, collate
    from tiny_voca.infer import match_frontend
    from tiny_voca.train import TrainingSystem, WEIGHTS
    from tiny_voca.config import Config
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    match_frontend(a.checkpoint)
    c = Config.from_dict(state["config"])
    system = TrainingSystem(c).to(a.device)
    system.model.load_state_dict(state["model"], strict=True)
    system.eval()
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        f = fingerprint_of(a.fingerprint, a.device)[None]
        start_token, start_frame = system.model.voices.project(f)[0], system.model.frame_voices.project(f)[0]
    token = torch.nn.Parameter(start_token.clone()); frame = torch.nn.Parameter(start_frame.clone())
    install(system.model, token, frame)
    data = AnnotatedDataset(a.annotations, c, cache=True)
    batch = collate([data[i] for i in range(len(data))], a.device)
    optimizer = torch.optim.Adam([token, frame], lr=a.learning_rate)
    first = None
    for step in range(1, a.steps + 1):
        loss, terms, _ = system.objective(batch, 0.5, WEIGHTS, random_phase=False)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        first = first if first is not None else float(loss)
        if step % 100 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": round(float(loss), 3), "mel": round(float(terms["mel"]), 4), "pitch": round(float(terms["pitch"]), 4)}), flush=True)
    torch.save({"token": token.detach().cpu(), "frame": frame.detach().cpu(), "checkpoint": a.checkpoint,
                "moved": {"token": float((token - start_token).norm() / start_token.norm()), "frame": float((frame - start_frame).norm() / start_frame.norm())}}, a.output)
    print(json.dumps({"saved": str(a.output), "loss": [round(first, 3), round(float(loss), 3)]}))


if __name__ == "__main__":
    main()
