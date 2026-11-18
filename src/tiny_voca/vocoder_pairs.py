"""Vocoder fine-tuning pairs: this voice model's own spectrogram for each recording. Local GPU or CPU.

    python -m tiny_voca.vocoder_pairs --checkpoint runs/voice-emotion/inference.pt \
        --annotations data/it-multi6-annotations/train.jsonl data/emotion-annotations/train.jsonl \
        --output data/vocoder-pairs/train

For every annotated clip the voice model runs with the recording's own durations,
pitch, voicing and energy, so its source-filter waveform lines up with the
recording sample for sample. Its Vocos log-mel (what the vocoder reads at
synthesis time) is stored as float16; ``train_vocoder --pairs`` then teaches the
vocoder to turn these spectrograms, artefacts included, into the real recording.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .data import AnnotatedDataset, collate
from .infer import load_model, match_frontend
from .losses import VocosMel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, help="Only the first N clips (smoke test)")
    args = parser.parse_args()
    match_frontend(str(args.checkpoint))
    model = load_model(str(args.checkpoint), args.device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    features = VocosMel().to(args.device)
    dataset = AnnotatedDataset([str(a) for a in args.annotations], model.config, cache=False)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "pairs.jsonl"
    done = {json.loads(l)["id"] for l in manifest.read_text().splitlines()} if manifest.exists() else set()
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    with open(manifest, "a") as out, torch.no_grad():
        for index in range(count):
            row = dataset.rows[index]
            if row["id"] in done:
                continue
            batch = collate([dataset[index]], args.device)
            emotions, intensities = batch.get("emotions"), batch.get("intensities")
            hidden, mask = model.encode(batch["tokens"], batch["stress"], batch["token_lengths"], batch["speakers"], emotions, intensities)
            frames, frame_mask, lengths = model.expand(hidden, batch["durations"], mask, batch["speakers"], emotions, intensities)
            prosody_hidden, _ = model.predict_prosody(frames, frame_mask)
            acoustic = model.decode(prosody_hidden, batch["prosody"], frame_mask)
            torch.manual_seed(index)
            wave = model.vocoder(acoustic, model.frequency(batch["prosody"][..., 0]), lengths)
            mel = features(wave)[0].half().cpu()                         # [frames + 1, bands]
            name = hashlib.sha256(row["id"].encode()).hexdigest()[:24] + ".pt"
            torch.save({"mel": mel}, args.output / name)
            real = (dataset.row_roots[index] / row["path"]).resolve()
            out.write(json.dumps({"id": row["id"], "speaker": row["speaker"], "path": str(real), "pair": str((args.output / name).resolve()),
                                  "frames": row["frames"], "voice_checkpoint_step": state.get("step")}) + "\n")
            if index % 500 == 0:
                out.flush()
                print(json.dumps({"done": index + 1, "of": count}), flush=True)
    print(json.dumps({"complete": True, "clips": count, "manifest": str(manifest)}), flush=True)


if __name__ == "__main__":
    main()
