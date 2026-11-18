"""Train the small neural vocoder (``tiny_voca.vocoder``) on recorded speech.

    python -m tiny_voca.train_vocoder --manifests data/it1-full-features/train.jsonl data/it-multi-features/train.jsonl \
        --exclude-speakers CML659 CML10446 CML2019 CML1725 --validation data/it1-full-features/test.jsonl \
        --output runs/vocoder --steps 500000

Reads the stored 24 kHz waveforms of prepared feature sets (no transcripts are
needed), trains on random crops with the Vocos/HiFi-GAN recipe: log-mel L1, the
multi-period and multi-resolution discriminators of ``losses`` (least squares)
and feature matching. Starts paid compute on a GPU; resumable with --resume.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .losses import Discriminators, VocosMel, discriminator_loss, generator_gan_loss
from .train import Throttle, atomic_save
from .vocoder import Vocoder, VocoderConfig


def rows(manifests, exclude=()):
    selected = []
    for manifest in manifests:
        root = Path(manifest).parent
        for line in Path(manifest).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("speaker") not in exclude:
                    item = {"path": str(root / row["path"]), "speaker": row.get("speaker"), "id": row["id"]}
                    if "pair" in row:
                        item["pair"] = row["pair"]
                    selected.append(item)
    return selected


class Crops(Dataset):
    """Random fixed-length crops of stored waveforms (short clips are zero-padded).

    Items with a ``pair`` (tiny_voca.vocoder_pairs) return, with probability
    ``pair_fraction``, the voice model's own log-mel for exactly the same frames as
    the input instead of the recording's: (wave, mel, 1). Otherwise (wave, zeros, 0)
    and the training loop computes the recording's mel itself."""

    def __init__(self, items, segment, pair_fraction=0.0, hop=256, bands=100):
        self.items, self.segment, self.pair_fraction, self.hop, self.bands = items, segment, pair_fraction, hop, bands

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        stored = torch.load(item["path"], map_location="cpu", weights_only=True)
        wave = stored["wave"].float()
        # Frame pitch in Hz and voicing for the harmonic prior (prepare stores log(F0 / 200) and a voiced flag).
        pitch = 200.0 * torch.exp(stored["prosody"][:, 0].float())
        voiced = (stored["prosody"][:, 1] > 0.5).float()
        frames = self.segment // self.hop
        empty = torch.zeros(frames + 1, self.bands)
        if wave.numel() <= self.segment:
            pad = frames - pitch.numel()
            return (torch.nn.functional.pad(wave, (0, self.segment - wave.numel())), empty, 0,
                    torch.nn.functional.pad(pitch, (0, pad), value=200.0), torch.nn.functional.pad(voiced, (0, pad)))
        start = random.randrange((wave.numel() - self.segment) // self.hop + 1)
        crop = wave[start * self.hop:start * self.hop + self.segment]
        f0, gate = pitch[start:start + frames], voiced[start:start + frames]
        if "pair" in item and random.random() < self.pair_fraction:
            mel = torch.load(item["pair"], map_location="cpu", weights_only=True)["mel"].float()
            if mel.shape[0] >= start + frames + 1:
                return crop, mel[start:start + frames + 1], 1, f0, gate
        return crop, empty, 0, f0, gate


def complex_distance(student, teacher, ffts=(512, 1024, 2048)):
    """Multi-resolution L1 between magnitude-compressed complex STFTs (|S|^0.3 with the
    phase kept): unlike a magnitude loss it also pulls the phase towards the teacher's."""
    total = 0.0
    for fft in ffts:
        window = torch.hann_window(fft, device=student.device)
        a = torch.stft(student, fft, fft // 4, window=window, return_complex=True)
        b = torch.stft(teacher, fft, fft // 4, window=window, return_complex=True)
        a = a * (a.abs() + 1e-7) ** -0.7
        b = b * (b.abs() + 1e-7) ** -0.7
        total = total + (a - b).abs().mean()
    return total / len(ffts)


def teacher_spectrum(teacher, mel):
    """Vocos' predicted (log magnitude, phase) [B, bins, frames] before its inverse STFT."""
    return teacher.head.out(teacher.backbone(mel)).transpose(1, 2).chunk(2, dim=1)


def spectrum_distance(student, teacher):
    """Frame-by-frame distance between two predicted spectra on the same STFT grid:
    L1 of the log magnitudes plus a magnitude-weighted 1 - cos(phase difference),
    which has no wrapping problem and useful gradients everywhere (APNet-style)."""
    (log_s, phase_s), (log_t, phase_t) = student, teacher
    floor, ceiling = math.log(1e-5), math.log(1e2)
    magnitude = (log_s.clamp(floor, ceiling) - log_t.clamp(floor, ceiling)).abs().mean()
    weight = torch.exp(log_t.clamp(floor, ceiling)) ** 0.3
    weight = weight / weight.mean(dim=(1, 2), keepdim=True)
    phase = (weight * (1 - torch.cos(phase_s - phase_t))).mean()
    return magnitude, phase


def load_teacher(name, device):
    """Pretrained Vocos 24 kHz (MIT), used only as a training target; needs the vocos package."""
    if name != "vocos":
        raise ValueError(f"unknown teacher {name!r}")
    from vocos import Vocos
    teacher = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def learning_rate(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    return 0.5 * base * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


@torch.no_grad()
def validate(vocoder, features, items, device):
    """Copy synthesis of whole utterances: mean log-mel L1 against the recording."""
    vocoder.eval()
    total = 0.0
    for item in items:
        stored = torch.load(item["path"], map_location="cpu", weights_only=True)
        wave = stored["wave"].float()[None].to(device)
        mel = features(wave).transpose(1, 2)
        f0 = (200.0 * torch.exp(stored["prosody"][:, 0].float()))[None].to(device)
        voiced = (stored["prosody"][:, 1] > 0.5).float()[None].to(device)
        torch.manual_seed(0)
        fake = vocoder(mel, f0, voiced)[:, :wave.shape[1]]
        total += float((features(fake) - features(wave[:, :fake.shape[1]])).abs().mean())
    vocoder.train()
    return total / len(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", nargs="+", required=True, help="Prepared feature manifests (train splits)")
    parser.add_argument("--exclude-speakers", nargs="*", default=[])
    parser.add_argument("--validation", required=True, help="Feature manifest for copy-synthesis validation")
    parser.add_argument("--validation-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--segment", type=int, default=16384, help="Crop length in samples (multiple of the hop)")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--gan-start", type=int, default=2000, help="Steps of mel-only training before the critics join")
    parser.add_argument("--mel-weight", type=float, default=45.0)
    parser.add_argument("--feature-matching", type=float, default=2.0)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--intermediate", type=int, default=576)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--keep-every", type=int, default=50000)
    parser.add_argument("--validate-every", type=int, default=5000)
    parser.add_argument("--throttle-file", type=Path)
    parser.add_argument("--resume", action="store_true", help="Continue from OUTPUT/last.pt")
    parser.add_argument("--init-from", type=Path, help="Start a new run from another run's last.pt (weights, critics, optimizers, step)")
    parser.add_argument("--teacher", choices=("vocos",), help="Distil from this pretrained vocoder on the same mel input")
    parser.add_argument("--distill-weight", type=float, default=0.0, help="Weight of the distance to the teacher")
    parser.add_argument("--distill-mode", choices=("spectrum", "wave"), default="spectrum",
                        help="spectrum: match the teacher's predicted magnitude and phase frame by frame (same STFT grid); "
                        "wave: multi-resolution complex-STFT distance between the two waveforms")
    parser.add_argument("--fresh-schedule", action="store_true", help="With --init-from: warm-up and cosine restart at the start step")
    parser.add_argument("--prior", action="store_true", help="Harmonic-prior vocoder: phase from a pulse train built from the pitch")
    parser.add_argument("--prior-high-hz", type=float, default=0.0, help="Weaken the prior's pulses above this frequency (0 = off)")
    parser.add_argument("--prior-high-gain", type=float, default=1.0, help="Pulse share at twice --prior-high-hz and above")
    parser.add_argument("--pairs", nargs="*", default=[], help="tiny_voca.vocoder_pairs manifests: the voice model's own spectrograms")
    parser.add_argument("--pair-fraction", type=float, default=0.7, help="Share of paired clips that use the voice model's spectrogram")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.segment % 256:
        parser.error("--segment must be a multiple of the hop (256)")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=True)

    config = VocoderConfig(channels=args.channels, intermediate=args.intermediate, layers=args.layers, prior=args.prior,
                           prior_high_hz=args.prior_high_hz, prior_high_gain=args.prior_high_gain)
    vocoder = Vocoder(config).to(args.device)
    critics = Discriminators().to(args.device)
    features = VocosMel(config.sample_rate, config.fft, config.hop, config.bands).to(args.device)
    optimizer_g = torch.optim.AdamW(vocoder.parameters(), lr=args.learning_rate, betas=(0.8, 0.9))
    optimizer_d = torch.optim.AdamW(critics.parameters(), lr=args.learning_rate, betas=(0.8, 0.9))
    step = 0
    origin = 0          # step at which this run's learning-rate schedule starts (a warm start begins a fresh schedule)
    last = args.output / "last.pt"
    if args.resume and args.init_from:
        parser.error("--init-from starts a new run; --resume continues one")
    if args.distill_weight > 0 and not args.teacher:
        parser.error("--distill-weight needs --teacher")
    if args.resume or args.init_from:
        state = torch.load(last if args.resume else args.init_from, map_location="cpu", weights_only=True)
        source = VocoderConfig.from_dict(state["config"])
        if args.resume and source != config:
            raise ValueError("resume configuration differs")
        same_weights = VocoderConfig(**{**source.to_dict(), "prior_high_hz": 0.0, "prior_high_gain": 1.0}) == \
            VocoderConfig(**{**config.to_dict(), "prior_high_hz": 0.0, "prior_high_gain": 1.0})
        if source == config or (args.init_from and same_weights):
            # The high-band pulse weight changes only the prior, not the weights: a branch keeps everything.
            vocoder.load_state_dict(state["vocoder"])
            optimizer_g.load_state_dict(state["optimizer_g"])
        elif args.init_from and source == VocoderConfig(**{**config.to_dict(), "prior": False}) and config.prior:
            # Warm start of a prior vocoder from a plain one: same backbone and magnitude head; the
            # phase rows now predict a correction to the prior's phase, so they start at zero.
            missing, unexpected = vocoder.load_state_dict(state["vocoder"], strict=False)
            if unexpected or any(not name.startswith("prior_in.") for name in missing):
                raise ValueError(f"cannot warm-start: missing {missing}, unexpected {unexpected}")
            bins = config.fft // 2 + 1
            with torch.no_grad():
                vocoder.head.weight[bins:].zero_()
                vocoder.head.bias[bins:].zero_()
        else:
            raise ValueError("--init-from checkpoint has a different vocoder configuration")
        critics.load_state_dict(state["critics"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        step = state["step"]
        origin = state.get("schedule_origin", 0) if args.resume else (step if args.fresh_schedule else 0)

    teacher = load_teacher(args.teacher, args.device) if args.teacher else None
    train_items = rows(args.pairs, set(args.exclude_speakers)) if args.pairs else rows(args.manifests, set(args.exclude_speakers))
    validation_items = rows([args.validation])[:args.validation_size]
    speakers = sorted({item["speaker"] for item in train_items})
    parameters = sum(p.numel() for p in vocoder.parameters())
    print(json.dumps({"parameters": parameters, "train_clips": len(train_items), "speakers": speakers, "start_step": step}), flush=True)
    (args.output / "config.json").write_text(json.dumps({"vocoder": config.to_dict(), "args": {k: str(v) for k, v in vars(args).items()},
                                                         "speakers": speakers, "train_clips": len(train_items)}, indent=2))
    loader = DataLoader(Crops(train_items, args.segment, args.pair_fraction if args.pairs else 0.0), batch_size=args.batch, drop_last=True, num_workers=args.workers,
                        persistent_workers=args.workers > 0, pin_memory=args.device.startswith("cuda"),
                        sampler=RandomSampler(train_items, replacement=True, num_samples=10 ** 9))
    throttle = Throttle(args.throttle_file)
    metrics = open(args.output / "metrics.jsonl", "a")
    window = {"mel": 0.0, "adversarial": 0.0, "matching": 0.0, "critics": 0.0, "distill": 0.0, "count": 0, "start": time.time(), "throttle": 0.0}

    def save(path):
        atomic_save({"config": config.to_dict(), "vocoder": vocoder.state_dict(), "critics": critics.state_dict(),
                     "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(), "step": step,
                     "schedule_origin": origin}, path)

    for real, paired_mel, paired, f0, voiced in loader:
        if step >= args.steps:
            break
        real = real.to(args.device, non_blocking=True)
        for optimizer in (optimizer_g, optimizer_d):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step - origin, args.learning_rate, args.warmup, args.steps - origin)
        real_mel = features(real)
        if bool(paired.any()):
            # The input is the voice model's spectrogram; the target stays the recording.
            use = paired.to(args.device).bool()
            input_mel = torch.where(use[:, None, None], paired_mel.to(args.device), real_mel)
        else:
            input_mel = real_mel
        prior = vocoder.analyse_prior(f0.to(args.device), voiced.to(args.device)) if config.prior else None
        predicted = vocoder.spectrum(input_mel.transpose(1, 2), prior)
        fake = vocoder.render(*predicted, prior=prior)[:, :real.shape[1]]
        adversarial_on = step >= args.gan_start
        if adversarial_on:
            d_loss = discriminator_loss(critics(real), critics(fake.detach()))
            optimizer_d.zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(critics.parameters(), 10.0)
            optimizer_d.step()
        mel = (features(fake) - real_mel).abs().mean()
        loss = args.mel_weight * mel
        if teacher is not None:
            # The target is always the teacher on the REAL spectrogram (clean), even for paired input.
            with torch.no_grad():
                if args.distill_mode == "spectrum":
                    target_spectrum = teacher_spectrum(teacher, real_mel.transpose(1, 2))
                else:
                    target = teacher.decode(real_mel.transpose(1, 2))[:, :fake.shape[1]]
            if args.distill_mode == "spectrum":
                magnitude_part, phase_part = spectrum_distance(predicted, target_spectrum)
                # With a prior, the phase comes from the pulse train: distil only the magnitude.
                distill = magnitude_part if config.prior else magnitude_part + phase_part
                window["distill_phase"] = window.get("distill_phase", 0.0) + float(phase_part.detach())
            else:
                distill = complex_distance(fake, target)
            loss = loss + args.distill_weight * distill
            window["distill"] += float(distill.detach())
        if adversarial_on:
            adversarial, matching = generator_gan_loss(critics(real), critics(fake))
            loss = loss + adversarial + args.feature_matching * matching
        optimizer_g.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vocoder.parameters(), 10.0)
        optimizer_g.step()
        step += 1
        window["mel"] += float(mel.detach())
        if adversarial_on:
            window["adversarial"] += float(adversarial.detach())
            window["matching"] += float(matching)
            window["critics"] += float(d_loss)
        window["count"] += 1
        window["throttle"] += throttle()
        if step % 100 == 0:
            elapsed = time.time() - window["start"]
            n = window["count"]
            record = {"step": step, "steps_per_second": n / elapsed, "mel": window["mel"] / n, "adversarial": window["adversarial"] / n,
                      "feature_matching": window["matching"] / n, "critics": window["critics"] / n,
                      "learning_rate": optimizer_g.param_groups[0]["lr"], "throttle_seconds": window["throttle"]}
            if teacher is not None:
                record["distill"] = window["distill"] / n
                if "distill_phase" in window:
                    record["distill_phase"] = window["distill_phase"] / n
            if step % args.validate_every == 0:
                record["validation_mel"] = validate(vocoder, features, validation_items, args.device)
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            window = {"mel": 0.0, "adversarial": 0.0, "matching": 0.0, "critics": 0.0, "distill": 0.0, "count": 0, "start": time.time(), "throttle": 0.0}
        if step % args.save_every == 0 or step == args.steps:
            save(last)
            atomic_save({"config": config.to_dict(), "vocoder": vocoder.state_dict(), "step": step}, args.output / "vocoder.pt")
            if step % args.keep_every == 0:
                atomic_save({"config": config.to_dict(), "vocoder": vocoder.state_dict(), "step": step},
                            args.output / f"vocoder-{step:07d}.pt")
    save(last)
    atomic_save({"config": config.to_dict(), "vocoder": vocoder.state_dict(), "step": step}, args.output / "vocoder.pt")
    print(json.dumps({"done": step}), flush=True)


if __name__ == "__main__":
    main()
