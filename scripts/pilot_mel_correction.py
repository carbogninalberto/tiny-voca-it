"""Pilot: train only a MelCorrection (about 29k params) on top of a frozen voice model.

    python scripts/pilot_mel_correction.py --checkpoint runs/voice-emotion/inference.pt \
        --train data/it-multi6-annotations/train.jsonl data/emotion-annotations/train.jsonl \
        --output runs/mel-correction-pilot --steps 5000

Per batch, as at synthesis time: recorded durations, the model's own predicted
prosody conditioning the acoustic stack, and the recorded pitch for the
source-filter waveform (so frames and harmonics line up with the recording). The
correction maps the Vocos mel of that waveform towards the recording's Vocos mel
(L1, valid frames only); the voice model itself is not changed.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_voca.data import AnnotatedDataset, BucketSampler, collate  # noqa: E402
from tiny_voca.infer import load_model, match_frontend  # noqa: E402
from tiny_voca.losses import VocosMel  # noqa: E402
from tiny_voca.model import MelCorrection  # noqa: E402
from tiny_voca.train import Throttle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--plateau-patience", type=int, default=0, help="Halve the learning rate after this many steps "
                        "without a new best 100-step loss (0 = off)")
    parser.add_argument("--low-hz", type=float, default=1000.0)
    parser.add_argument("--batch-frames", type=int, default=9000)
    parser.add_argument("--throttle-file", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=1000, help="Save correction.pt every N steps (the playground reloads it)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    match_frontend(args.checkpoint)
    model = load_model(args.checkpoint, args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    c = model.config
    correction = MelCorrection(c.channels, low_hz=args.low_hz).to(args.device)
    features = VocosMel().to(args.device)
    optimizer = torch.optim.AdamW(correction.parameters(), lr=args.learning_rate, weight_decay=0.0)
    data = AnnotatedDataset(args.train, c, cache=False)
    sampler = BucketSampler(data.lengths, args.batch_frames, 32, seed=0)
    throttle = Throttle(args.throttle_file)
    log = open(args.output / "metrics.jsonl", "a")
    start, window = time.time(), []
    plateau_scale, best, best_step = 1.0, float("inf"), 0

    def save(step):
        temporary = args.output / "correction.pt.tmp"
        torch.save({"correction": correction.state_dict(), "channels": c.channels, "low_hz": args.low_hz,
                    "voice_checkpoint": args.checkpoint, "steps": step}, temporary)
        temporary.replace(args.output / "correction.pt")
    for step in range(1, args.steps + 1):
        # Warm-up, cosine to 1% of the peak, and an adaptive cut on a plateau.
        cosine = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        rate = args.learning_rate * min(1.0, step / args.warmup) * cosine * plateau_scale
        for group in optimizer.param_groups:
            group["lr"] = rate
        batch = collate([data[i] for i in sampler.next()], args.device)
        with torch.no_grad():
            emotions, intensities = batch.get("emotions"), batch.get("intensities")
            hidden, mask = model.encode(batch["tokens"], batch["stress"], batch["token_lengths"], batch["speakers"], emotions, intensities)
            frames, frame_mask, lengths = model.expand(hidden, batch["durations"], mask, batch["speakers"], emotions, intensities)
            prosody_hidden, prediction = model.predict_prosody(frames, frame_mask)
            acoustic = model.decode(prosody_hidden, model.hard_prosody(prediction.float()), frame_mask)
            wave = model.vocoder(acoustic, model.frequency(batch["prosody"][..., 0]), lengths)
            dsp_mel, real_mel = features(wave), features(batch["wave"])
            valid = (torch.arange(dsp_mel.shape[1], device=args.device)[None] <= lengths[:, None]).float()[..., None]
            before = float(((dsp_mel - real_mel).abs() * valid * correction.mask).sum() / (valid.sum() * correction.mask.sum()))
        corrected = correction(acoustic, dsp_mel)
        loss = ((corrected - real_mel).abs() * valid * correction.mask).sum() / (valid.sum() * correction.mask.sum())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        throttle()
        window.append((float(loss), before))
        if step % 100 == 0:
            average = sum(w[0] for w in window) / len(window)
            if average < best - 1e-4:
                best, best_step = average, step
            elif args.plateau_patience and step - best_step >= args.plateau_patience and step > args.warmup:
                plateau_scale, best_step = plateau_scale * 0.5, step
            record = {"step": step, "learning_rate": rate, "loss_above_1k": average,
                      "uncorrected_above_1k": sum(w[1] for w in window) / len(window),
                      "steps_per_second": step / (time.time() - start)}
            log.write(json.dumps(record) + "\n"); log.flush(); window = []
            print(json.dumps(record), flush=True)
        if step % args.save_every == 0 and step < args.steps:
            save(step)
    save(args.steps)
    print(json.dumps({"done": args.steps}), flush=True)


if __name__ == "__main__":
    main()
