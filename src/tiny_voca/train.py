"""From-scratch training on validated durations. Running this starts paid compute.

There is no learned aligner and no alignment loss: token durations come from
the annotation layer and must cover the acoustic frames exactly. The waveform
discriminators and the auxiliary mel head are training-only. No model or data
downloads and no infrastructure management happen here.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from .config import CHECKPOINT_FORMAT, Config
from .data import AnnotatedDataset, BucketSampler, collate
from .journal import file_hash
from .losses import Discriminators, SpectralLoss, VocosMel, discriminator_loss, generator_gan_loss, masked_mean
from .model import TinyVoca
from .text import frontend_fingerprint, frontend_identity

# Custom-voice adaptation: which parts of the network a mode trains. Everything
# else is frozen (requires_grad False AND left out of the optimizer, so AdamW
# decay/momentum cannot move it either).
ADAPT_MODES = {
    "embedding": ("model.voices.", "model.frame_voices."),
    "acoustic": ("model.voices.", "model.frame_voices.", "model.duration.", "model.frame_position.", "model.prosody_stack.",
                 "model.prosody_head.", "model.prosody_projection.", "model.acoustic_stack.", "model.vocoder.", "model.mel_output.",
                 "mel_head."),
    "full": ("",),
}
SPEAKER_TABLES = ("model.voices.weight", "model.frame_voices.weight")

VOCOS_MEL_WEIGHT = 2.0
WEIGHTS = {"duration": 1.0, "pause": 1.0, "pitch": 2.0, "pitch_spread": 2.0, "voicing": 1.0, "energy": 1.0, "mel_head": 2.0,
           "stft": 2.0, "mel": 10.0}
SEMITONES = 12 / math.log(2)


class Throttle:
    """Pause after every step for the time given in a JSON file ({"sleep": s}),
    re-read every few seconds, so an outside monitor can cap GPU load and power
    while training runs. Pauses never change what is computed."""

    def __init__(self, path, every=5.0):
        self.path, self.every, self.sleep, self.checked = path, every, 0.0, -math.inf

    def __call__(self):
        if self.path is None:
            return 0.0
        now = time.monotonic()
        if now - self.checked >= self.every:
            self.checked = now
            try:
                self.sleep = min(5.0, max(0.0, float(json.loads(Path(self.path).read_text())["sleep"])))
            except (OSError, ValueError, KeyError, TypeError):
                pass        # keep the last valid value while the file is rewritten
        if self.sleep > 0:
            time.sleep(self.sleep)
        return self.sleep


def masked_std(value, mask):
    """Standard deviation per row over the masked positions, [B]."""
    count = mask.sum(1).clamp_min(1)
    mean = (value * mask).sum(1) / count
    return (((value - mean[:, None]).square() * mask).sum(1) / count + 1e-8).sqrt()


class TrainingSystem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = TinyVoca(config)
        self.mel_head = nn.Linear(config.channels, config.n_mels)   # auxiliary, not exported
        self.spectral = SpectralLoss(config.sample_rate)
        self.vocos_mel = VocosMel(bands=config.vocos_mels or 100) if config.vocos_mels or config.vocos_mel_loss else None

    _texture_bands = {}

    @classmethod
    def texture_bands(cls, bands, device, low_hz=2000.0):
        """Indices of the Vocos mel bands centred at or above ``low_hz`` (cached)."""
        key = (bands, str(device), low_hz)
        if key not in cls._texture_bands:
            from .losses import VocosMel
            centre = VocosMel().filters.argmax(1).float() * 24000 / 1024
            cls._texture_bands[key] = (centre >= low_hz).nonzero()[:, 0].to(device)
        return cls._texture_bands[key]

    def objective(self, batch, teacher_probability, weights=WEIGHTS, random_phase=True, pitch_delta=0.0, texture=0.0):
        """Loss on full utterances. Returns ``(loss, terms, auxiliary)``.

        ``teacher_probability`` is the per-utterance chance that the acoustic
        decoder is conditioned on recorded pitch/voicing/energy; otherwise it
        gets the model's own (detached) predictions, exactly as at inference.
        The oscillator always follows the recorded pitch here, because the
        waveform target does.
        """
        m, c = self.model, self.model.config
        hidden, token_mask = m.encode(batch["tokens"], batch["stress"], batch["token_lengths"], batch["speakers"], batch.get("emotions"), batch.get("intensities"))
        output = m.duration(hidden, token_mask).float()
        # Length is learned only where a token has frames; whether a pause-capable
        # token has any is a separate yes/no target.
        present = batch["durations"] > 0
        duration = masked_mean((output[..., 0] - torch.log1p(batch["durations"].float())).square(), token_mask & present)
        pausable = token_mask & ~m.phone_tokens[batch["tokens"]]
        pause = masked_mean(F.binary_cross_entropy_with_logits(output[..., 1], present.float(), reduction="none"), pausable)
        frames, frame_mask, lengths = m.expand(hidden, batch["durations"], token_mask, batch["speakers"], batch.get("emotions"), batch.get("intensities"))
        if not torch.equal(lengths, batch["frame_lengths"]):
            raise ValueError("durations do not sum to the acoustic frame count")
        prosody_hidden, prediction = m.predict_prosody(frames, frame_mask)
        prediction, target = prediction.float(), batch["prosody"]
        trusted = batch["pitch_trusted"]
        # Recorded pitch counts fully; interpolated stretches only keep the curve continuous.
        pitch = masked_mean((prediction[..., 0] - target[..., 0]).abs(), frame_mask * (trusted + 0.1 * (1 - trusted)))
        # An absolute-error fit alone predicts a flatter contour than people speak
        # (about three quarters of the recorded spread on held-out text). Match each
        # utterance's pitch spread as well, the classic global-variance correction.
        sung = frame_mask * target[..., 1] * trusted
        enough = (sung.sum(1) >= 10).float()
        predicted_spread, recorded_spread = masked_std(prediction[..., 0], sung), masked_std(target[..., 0], sung)
        pitch_spread = masked_mean((predicted_spread - recorded_spread).abs(), enough)
        voicing = masked_mean(F.binary_cross_entropy_with_logits(prediction[..., 1], target[..., 1], reduction="none"), frame_mask)
        energy = masked_mean((prediction[..., 2] - target[..., 2]).abs(), frame_mask)
        teacher = torch.rand(len(lengths), device=lengths.device) < teacher_probability
        condition = torch.where(teacher[:, None, None], target, m.hard_prosody(prediction.detach()))
        acoustic = m.decode(prosody_hidden, condition.to(prosody_hidden.dtype), frame_mask)
        mel_target = (batch["mel"] - c.mel_mean) / c.mel_scale
        mel_head = masked_mean((self.mel_head(acoustic).float() - mel_target).abs().mean(-1), frame_mask)
        phase = 2 * math.pi * torch.rand(len(lengths), device=lengths.device) if random_phase else None
        wave = m.vocoder(acoustic, m.frequency(target[..., 0]), lengths, phase=phase, voiced=target[..., 1])
        stft, mel = self.spectral(wave, batch["wave"], lengths * c.hop)
        terms = dict(duration=duration, pause=pause, pitch=pitch, pitch_spread=pitch_spread, voicing=voicing, energy=energy,
                     mel_head=mel_head, stft=stft, mel=mel)
        loss = sum(weights[name] * value for name, value in terms.items())
        # Frame-to-frame pitch change against the recording's, on trusted voiced frame pairs. The spread
        # loss alone can be met by frame-level wiggle (heard as a tremble); this asks for the recorded
        # movement instead. Always measured; weighted only when the recipe sets ``pitch_delta``.
        pair = sung[:, 1:] * sung[:, :-1]
        terms["pitch_delta"] = masked_mean(((prediction[:, 1:, 0] - prediction[:, :-1, 0]) - (target[:, 1:, 0] - target[:, :-1, 0])).abs(), pair)
        if pitch_delta:
            loss = loss + pitch_delta * terms["pitch_delta"]
        if c.vocos_mel_loss:
            frames_vocos = self.vocos_mel(batch["wave"]).shape[1]
            vocos_mask = torch.arange(frames_vocos, device=lengths.device)[None] <= lengths[:, None]
            with torch.no_grad():
                real_vocos = self.vocos_mel(batch["wave"])
            predicted_vocos = self.vocos_mel(wave)
            terms["vocoder_input"] = masked_mean((predicted_vocos - real_vocos).abs().mean(-1), vocos_mask)
            loss = loss + c.vocos_mel_loss * terms["vocoder_input"]
            # Texture above 2 kHz: match the AMOUNT of frame-to-frame change per band and utterance, not the
            # values. Point-wise L1 rewards the average and smooths noise-like detail away (model output has
            # ~0.88x the recorded detail there); like the pitch-spread loss, this asks for the recorded
            # variability. Always measured; weighted only when the recipe sets ``texture``.
            high = self.texture_bands(predicted_vocos.shape[-1], predicted_vocos.device)
            pair = (vocos_mask[:, 1:] & vocos_mask[:, :-1]).float()[..., None]
            change = lambda x: ((x[:, 1:, high] - x[:, :-1, high]).abs() * pair).sum(1) / pair.sum(1).clamp_min(1)
            terms["texture"] = (change(predicted_vocos) - change(real_vocos)).abs().mean()
            if texture:
                loss = loss + texture * terms["texture"]
        if m.mel_output is not None:
            # Kept out of WEIGHTS so existing recipes (and their resume checks) are unchanged.
            with torch.no_grad():
                vocos_target = self.vocos_mel(batch["wave"])[:, :frame_mask.shape[1]]
            terms["vocos_mel"] = masked_mean((m.mel_output(acoustic).float() - vocos_target).abs().mean(-1), frame_mask)
            loss = loss + VOCOS_MEL_WEIGHT * terms["vocos_mel"]
        with torch.no_grad():
            voiced = sung
            predicted = m.predict_durations(hidden, token_mask, batch["tokens"])
            predicted_frames = predicted.sum(1)
            pause_frames = (batch["durations"] * pausable).sum().clamp_min(1)
            terms.update(
                base_loss=loss.detach(), teacher_fraction=teacher.float().mean(),
                pitch_error_semitones=SEMITONES * masked_mean((prediction[..., 0] - target[..., 0]).abs(), voiced),
                pitch_spread_ratio=masked_mean(predicted_spread / recorded_spread.clamp_min(1e-3), enough),
                voicing_accuracy=masked_mean(((prediction[..., 1] > 0) == target[..., 1].bool()).float(), frame_mask),
                pause_accuracy=masked_mean(((output[..., 1] > 0) == present).float(), pausable),
                pause_length_ratio=(predicted * pausable).sum() / pause_frames,
                duration_length_ratio=(predicted_frames / lengths).mean(),
                duration_length_error=((predicted_frames - lengths).abs() / lengths).mean())
        return loss, terms, {"wave": wave, "reconstruction": weights["stft"] * stft + weights["mel"] * mel}


def crops(fake, real, lengths, size, count):
    """Aligned random windows [B*count, size] from generated and recorded audio,
    plus the sample indices they cover ([B, count*size])."""
    size = min(size, int(lengths.min()))
    start = (torch.rand(len(lengths), count, device=fake.device) * (lengths[:, None] - size + 1)).long()
    index = (start[..., None] + torch.arange(size, device=fake.device)).flatten(1)
    return fake.gather(1, index).view(-1, size), real.gather(1, index).view(-1, size), index


SCHEDULES = ("half_life", "cosine")
# Recipe fields added after format 4. A run recorded without them used these
# values, so resuming it compares equal and behaves exactly as before.
RECIPE_DEFAULTS = {"schedule": "half_life", "lr_floor": 0.0, "ema_decay": None, "total_steps": None, "mel_critic": False, "pitch_delta": 0.0, "texture": 0.0}


def learning_rate(step, base, warmup, half_life, start=0, schedule="half_life", total=None, floor=0.0):
    """Linear warm-up from ``start`` (0, or the step a run was branched at), then
    either halving every ``half_life`` steps or a cosine from ``base`` down to
    ``floor * base`` at step ``total`` (the end of this run)."""
    ramp = min(1.0, (step - start + 1) / max(warmup, 1))
    if schedule == "half_life":
        return base * ramp * 0.5 ** (step / half_life)
    if schedule != "cosine" or total is None or total <= start:
        raise ValueError("cosine schedule needs the run's final step")
    progress = min(1.0, max(0.0, (step - start) / (total - start)))
    return base * ramp * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


class EMA:
    """Exponential moving average of the inference model's weights (float32).

    Training-only. The decay ramps up as (1 + n) / (10 + n) over the first
    updates so that early, still-random weights are forgotten quickly. Buffers
    are copied, not averaged. The exported inference checkpoint uses these
    weights when the run has an EMA.
    """

    def __init__(self, model, decay, updates=0):
        self.decay, self.updates = decay, updates
        self.shadow = {k: v.detach().float().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for name, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[name].lerp_(value.detach().float(), 1 - decay)
            else:
                self.shadow[name].copy_(value)

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state):
        if set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA tensors do not match the model")
        self.decay, self.updates = state["decay"], state["updates"]
        self.shadow = {k: v.float().clone() for k, v in state["shadow"].items()}

    def weights_for(self, model):
        """The shadow in the model's own dtypes, ready for load_state_dict."""
        current = model.state_dict()
        return {k: v.to(dtype=current[k].dtype, device=current[k].device) for k, v in self.shadow.items()}


def swapped_weights(model, weights):
    """Context manager: run with ``weights`` loaded, then restore the originals exactly."""
    from contextlib import contextmanager

    @contextmanager
    def manager():
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(weights)
        try:
            yield model
        finally:
            model.load_state_dict(original)
    return manager()


def teacher_probability(step, hold, ramp, final):
    """1 while the prosody predictor is still poor, then down to ``final``."""
    return 1.0 if step < hold else max(final, 1.0 - (1.0 - final) * (step - hold) / max(ramp, 1))


def optimize_step(system, critics, optimizer, critic_optimizer, batch, step, recipe, parameters=None):
    weights = recipe["weights"]
    start = (recipe.get("branched_from") or {}).get("step", 0)
    schedule = {"schedule": recipe.get("schedule", "half_life"), "total": recipe.get("total_steps"), "floor": recipe.get("lr_floor", 0.0)}
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(step, recipe["learning_rate"], recipe["warmup"], recipe["lr_half_life"], start, **schedule)
    for group in critic_optimizer.param_groups:
        group["lr"] = learning_rate(step, recipe["learning_rate"], 1, recipe["lr_half_life"], start, **schedule)
    optimizer.zero_grad(set_to_none=True)
    probability = teacher_probability(step, recipe["teacher_hold"], recipe["teacher_ramp"], recipe["teacher_final"])
    loss, metrics, auxiliary = system.objective(batch, probability, weights, pitch_delta=recipe.get("pitch_delta", 0.0), texture=recipe.get("texture", 0.0))
    if step >= recipe["gan_start"]:
        wave = auxiliary["wave"]
        fake, real, index = crops(wave, batch["wave"], batch["frame_lengths"] * system.model.config.hop, recipe["crop_samples"], recipe["crops"])
        critic_optimizer.zero_grad(set_to_none=True)
        d_loss = discriminator_loss(critics(real), critics(fake.detach()))
        d_loss.backward()
        nn.utils.clip_grad_norm_(critics.parameters(), 10, error_if_nonfinite=True)
        critic_optimizer.step()
        critics.requires_grad_(False)
        try:
            with torch.no_grad():
                real_features = critics(real)
            adversarial, matching = generator_gan_loss(real_features, critics(fake))
            gan = adversarial + recipe["feature_matching"] * matching
            # Fixed weights cannot balance these terms: measured at the waveform, the
            # adversarial gradient was under 1% of the spectral one, i.e. switched off.
            # Scale the GAN term so that, inside the windows the critics saw, its
            # gradient is ``gan_ratio`` times the reconstruction gradient.
            reconstruction_gradient = torch.autograd.grad(auxiliary["reconstruction"], wave, retain_graph=True)[0].gather(1, index)
            gan_gradient = torch.autograd.grad(gan, wave, retain_graph=True)[0].gather(1, index)
            ramp = min(1.0, (step - recipe["gan_start"] + 1) / max(recipe["gan_ramp"], 1))
            scale = (recipe["gan_ratio"] * ramp * reconstruction_gradient.norm() / gan_gradient.norm().clamp_min(1e-12))
            scale = scale.clamp(max=recipe["gan_scale_max"]).detach()
            loss = loss + scale * gan
            metrics.update(discriminator=d_loss, adversarial=adversarial, feature_matching=matching, gan_scale=scale)
            loss.backward()
        finally:
            critics.requires_grad_(True)
    else:
        loss.backward()
    grad = nn.utils.clip_grad_norm_(system.parameters() if parameters is None else parameters, recipe["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    return {**{k: float(v.detach()) for k, v in metrics.items()}, "loss": float(loss.detach()), "grad_norm": float(grad),
            "frames": int(batch["frame_lengths"].sum()), "utterances": len(batch["frame_lengths"])}


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def export_model(model, path, step, annotation=None, provenance=None):
    profile = model.config.profile
    state = {"format_version": CHECKPOINT_FORMAT, "config": model.config.to_dict(), "step": step,
             "frontend": {**frontend_identity(profile), "fingerprint": frontend_fingerprint(profile)}, "annotation": annotation,
             "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if profile.language != "en":
        state["language"] = {"language": profile.language, "locale": profile.locale, "profile_version": profile.version,
                             "profile_digest": profile.digest}
    if provenance:
        state["provenance"] = provenance
    atomic_save(state, path)


def adaptation_init(system, source, init_voice):
    """Initialise a NEW custom-voice run from a trained Italian (or any) checkpoint.

    ``source`` is a training snapshot (``last.pt``: has ``system``) or an
    inference checkpoint (``inference.pt``: has ``model``). Every non-speaker
    tensor must exist with the same shape and is copied exactly. Both speaker
    tables are migrated BY NAME: a target voice that exists in the source keeps
    its row; a new one starts from the row of ``init_voice``. Returns a report
    of copied / initialised / fresh tensors; anything else is an error.
    """
    source_config = source["config"]
    target = system.model.config
    weights = source["system"] if "system" in source else {"model." + k: v for k, v in source["model"].items()}
    current = system.state_dict()
    report = {"copied": [], "speaker_rows": {}, "fresh": [], "rejected": []}
    source_speakers = list(source_config["speakers"])
    if init_voice not in source_speakers:
        raise ValueError(f"--init-voice {init_voice!r} is not a voice of the source checkpoint {source_speakers}")
    for name, tensor in current.items():
        if name in SPEAKER_TABLES:
            if name not in weights or weights[name].shape[1:] != tensor.shape[1:]:
                report["rejected"].append(name)
                continue
            rows = tensor.clone()
            for index, voice in enumerate(target.speakers):
                origin = voice if voice in source_speakers else init_voice
                rows[index] = weights[name][source_speakers.index(origin)]
                report["speaker_rows"].setdefault(voice, origin)
            current[name] = rows
        elif name in weights:
            if weights[name].shape != tensor.shape:
                report["rejected"].append(name)
            else:
                current[name] = weights[name]
                report["copied"].append(name)
        elif name.startswith("mel_head."):
            report["fresh"].append(name)          # inference checkpoints carry no auxiliary head
        else:
            report["rejected"].append(name)
    extra = sorted(set(weights) - set(current))
    if report["rejected"] or extra:
        raise ValueError(f"cannot adapt: tensors differ (missing/shape: {report['rejected']}, unexpected: {extra[:10]})")
    system.load_state_dict(current)
    report["copied"] = len(report["copied"])
    return report


def adaptation_parameters(system, mode):
    """Freeze everything outside the mode; return the trainable parameters."""
    prefixes = ADAPT_MODES[mode]
    trainable = []
    for name, parameter in system.named_parameters():
        keep = any(name.startswith(prefix) for prefix in prefixes)
        parameter.requires_grad_(keep)
        if keep:
            trainable.append(parameter)
    return trainable


def load_compatible(module, state):
    """Load weights for a branched run. A tensor that grew (same rank, every
    dimension at least as large) keeps the old values in its leading block and
    its fresh initialisation elsewhere. Returns what was not an exact copy."""
    current = module.state_dict()
    report = {"grown": [], "new": [], "dropped": sorted(set(state) - set(current))}
    for name, tensor in current.items():
        if name not in state:
            report["new"].append(name)
        elif state[name].shape == tensor.shape:
            current[name] = state[name]
        elif state[name].ndim == tensor.ndim and all(a <= b for a, b in zip(state[name].shape, tensor.shape)):
            merged = tensor.clone()
            merged[tuple(slice(0, size) for size in state[name].shape)] = state[name]
            current[name] = merged
            report["grown"].append(name)
        else:
            raise ValueError(f"cannot branch: {name} changed shape from {tuple(state[name].shape)} to {tuple(tensor.shape)}")
    module.load_state_dict(current)
    return report


def balanced_indices(dataset, limit):
    """Stable round-robin over voices, independent of manifest order."""
    groups = defaultdict(list)
    for i, row in enumerate(dataset.rows):
        groups[row["speaker"]].append(i)
    limit = min(max(limit, len(groups)), len(dataset.rows))
    selected = []
    for offset in range(limit):
        for voice in sorted(groups):
            if offset < len(groups[voice]):
                selected.append(groups[voice][offset])
                if len(selected) == limit:
                    return selected
    return selected


def check_leakage(train, validation):
    ids = {r["id"] for r in train.rows}
    texts = defaultdict(list)
    for row in train.rows:
        texts[row["text_hash"]].append(row["id"])
    collisions = []
    for row in validation.rows:
        if row["id"] in ids or row["text_hash"] in texts:
            collisions.append({"validation_id": row["id"], "train_ids": texts.get(row["text_hash"], [])[:3], "text": row.get("text", "")[:160]})
            if len(collisions) == 5:
                break
    if collisions:
        raise ValueError("train/validation leakage; split by sentence/text_id across BOTH voices. Examples: " + json.dumps(collisions, ensure_ascii=False))


@torch.no_grad()
def validation_batches(order, lengths, batch_size, frame_budget=None):
    """Consecutive batches of the length-sorted ``order`` with at most
    ``batch_size`` utterances and, if given, ``frame_budget`` padded frames."""
    batches, current = [], []
    for index in order:
        if current and (len(current) == batch_size or (frame_budget and (len(current) + 1) * lengths[index] > frame_budget)):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


def validate(system, dataset, indices, device, weights=WEIGHTS, batch_size=8, frame_budget=None):
    """Held-out metrics on a fixed balanced subset.

    ``teacher_forced`` conditions the decoder on recorded prosody, ``predicted``
    on the model's own; both use recorded durations and pitch for the oscillator
    so the spectra are comparable with the recording. Durations, pauses, pitch
    and voicing errors are those of the free-running predictors. This is a
    training monitor; pronunciation is judged separately with ASR and by listening.
    """
    system.eval()
    order = sorted(indices, key=lambda i: dataset.lengths[i])
    totals, weight = defaultdict(float), 0
    for chosen in validation_batches(order, dataset.lengths, batch_size, frame_budget):
        batch = collate([dataset[i] for i in chosen], device)
        size = len(batch["frame_lengths"])
        for name, probability in (("teacher_forced", 1.0), ("predicted", 0.0)):
            _, metrics, _ = system.objective(batch, probability, weights, random_phase=False)
            totals[f"validation_{name}_mel"] += float(metrics["mel"]) * size
            totals[f"validation_{name}_stft"] += float(metrics["stft"]) * size
            if "vocoder_input" in metrics:
                totals[f"validation_{name}_vocoder_input"] += float(metrics["vocoder_input"]) * size
            if "texture" in metrics:
                totals[f"validation_{name}_texture"] += float(metrics["texture"]) * size
        for key in ("duration", "pause_accuracy", "pause_length_ratio", "pitch_error_semitones", "pitch_spread_ratio", "pitch_delta", "voicing_accuracy",
                    "duration_length_ratio", "duration_length_error"):
            totals["validation_" + key] += float(metrics[key]) * size
        weight += size
    system.train()
    return {key: value / weight for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True, help="Annotation manifest(s) (train.jsonl); several are combined")
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-frames", type=int, default=9000, help="Padded acoustic frames per batch")
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--lr-half-life", type=float, default=100000)
    parser.add_argument("--lr-schedule", choices=SCHEDULES, default="half_life",
                        help="half_life: halve every --lr-half-life steps; cosine: decay to --lr-floor x the base rate at --steps")
    parser.add_argument("--lr-floor", type=float, default=0.0, help="Cosine schedule: final rate as a fraction of the base rate")
    parser.add_argument("--ema-decay", type=float, help="Keep an exponential moving average of the weights (e.g. 0.999) "
                        "and export it as inference.pt; validation then measures the EMA weights")
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--weights", type=json.loads, default={}, help="JSON overrides for the loss weights, e.g. '{\"mel\": 5}'")
    parser.add_argument("--gan-start", type=int, default=8000)
    parser.add_argument("--gan-ratio", type=float, default=0.7, help="Adversarial gradient as a fraction of the reconstruction gradient at the waveform")
    parser.add_argument("--gan-ramp", type=int, default=3000, help="Steps over which the adversarial term fades in while the critics warm up")
    parser.add_argument("--feature-matching", type=float, default=30.0, help="Feature-matching weight inside the adversarial term")
    parser.add_argument("--gan-scale-max", type=float, default=2000.0)
    parser.add_argument("--crop-samples", type=int, default=8192)
    parser.add_argument("--mel-critic", action="store_true", help="Add a critic on the Vocos log-mel above 1 kHz (what the neural vocoder reads)")
    parser.add_argument("--texture", type=float, default=0.0, help="Weight of the texture loss above 2 kHz (variability of the Vocos mel, 0 = off)")
    parser.add_argument("--pitch-delta", type=float, default=0.0, help="Weight of the frame-to-frame pitch-change loss (against tremble; 0 = off)")
    parser.add_argument("--crops", type=int, default=2, help="Discriminator windows per utterance")
    parser.add_argument("--teacher-hold", type=int, default=3000, help="Steps with recorded prosody conditioning only")
    parser.add_argument("--teacher-ramp", type=int, default=7000)
    parser.add_argument("--teacher-final", type=float, default=0.0, help="Share of utterances still conditioned on recorded prosody "
                        "after the ramp (0 = the decoder sees only what it gets at inference)")
    parser.add_argument("--subset", type=int, help="Train on the first N utterances (balanced by voice): memorisation check")
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-every", type=int, default=10000, help="Keep a numbered checkpoint at this interval and at the GAN boundary")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--max-seconds", type=float, default=3600, help="Wall-time cap per invocation; save and exit at a step boundary")
    parser.add_argument("--deadline-file", type=Path, help="Budget JSON with an absolute 'deadline'; training saves and exits before it")
    parser.add_argument("--deadline-margin", type=float, default=300)
    parser.add_argument("--resume", type=Path, help="Continue the same run exactly (same recipe, data and configuration)")
    parser.add_argument("--init-from", type=Path, help="Start a NEW run from another run's checkpoint with a changed recipe: "
                        "weights, step counter and already-trained critics are taken over; the data order starts fresh")
    parser.add_argument("--new-data", action="store_true", help="With --init-from: the checkpoint comes from a different annotation "
                        "run or front end (same symbol inventory)")
    parser.add_argument("--stop-file", type=Path, help="Save and exit after a step when this file appears")
    parser.add_argument("--recorded-voicing", action="store_true", help="Use the pitch tracker's voicing as it is, "
                        "without marking vowel frames voiced")
    parser.add_argument("--no-preload", action="store_true", help="Load feature files lazily instead of before the first step")
    parser.add_argument("--stream", action="store_true", help="Do not keep training features in memory: read each batch from disk, "
                        "the next one in background threads (for corpora larger than RAM). The data seen is unchanged")
    parser.add_argument("--throttle-file", type=Path, help='JSON {"sleep": seconds} re-read during training; that pause follows '
                        "every step (see scripts/gpu_governor.py)")
    parser.add_argument("--validation-frames", type=int, help="Padded-frame budget per validation batch (default: --batch-frames)")
    parser.add_argument("--adapt-from", type=Path, help="Start a NEW custom-voice run from a trained checkpoint (last.pt or "
                        "inference.pt): speaker tables migrated by name, fresh optimizer, critics and step counter")
    parser.add_argument("--adapt-mode", choices=sorted(ADAPT_MODES), default="acoustic",
                        help="embedding: only the voice rows; acoustic: voice rows + frame/prosody/acoustic/synthesis "
                        "(text encoder frozen); full: everything")
    parser.add_argument("--init-voice", help="Source voice whose rows initialise target voices that are new")
    parser.add_argument("--speaker-fingerprints", type=Path,
                        help="Fingerprint-voice models (speaker_embedding_dim > 0): torch file {'fingerprints': {voice: vector}}")
    parser.add_argument("--identity-new-blocks", action="store_true",
                        help="With --init-from into a deeper model: new residual blocks start as identity (output projections at zero)")
    parser.add_argument("--new-voice-init", nargs="*", default=[], metavar="NEW=SOURCE",
                        help="With --init-from: start an appended voice's speaker rows from an existing voice (e.g. EM4=EM1)")
    args = parser.parse_args()
    if min(args.steps, args.batch_frames, args.max_batch, args.save_every, args.keep_every, args.log_every, args.crops, args.gan_ramp) < 1:
        parser.error("steps, batch limits and intervals must be positive")
    if args.max_seconds <= 0 or args.learning_rate <= 0 or args.lr_half_life <= 0 or not 0 <= args.teacher_final <= 1:
        parser.error("time cap, learning rate and half life must be positive; teacher_final in [0, 1]")
    if set(args.weights) - set(WEIGHTS) or min(args.gan_ratio, args.feature_matching, args.gan_scale_max) < 0:
        parser.error(f"--weights may only override {sorted(WEIGHTS)}; GAN settings must be non-negative")
    if args.resume and args.init_from:
        parser.error("--init-from starts a new run; --resume continues one")
    if args.adapt_from and (args.init_from or args.resume):
        parser.error("--adapt-from starts a new adaptation run; resume it with --resume alone")
    if not 0 <= args.lr_floor < 1 or (args.ema_decay is not None and not 0 < args.ema_decay < 1):
        parser.error("--lr-floor must be in [0, 1) and --ema-decay in (0, 1)")

    # The budget file is re-read while training so a deliberate extension (or cut) takes effect.
    budget = {"deadline": math.inf, "checked": time.time()}
    if args.deadline_file:
        budget["deadline"] = float(json.loads(args.deadline_file.read_text())["deadline"]) - args.deadline_margin

    def deadline():
        if args.deadline_file and time.time() - budget["checked"] > 30:
            budget["checked"] = time.time()
            try:
                budget["deadline"] = float(json.loads(args.deadline_file.read_text())["deadline"]) - args.deadline_margin
            except (OSError, ValueError, KeyError):
                pass    # keep the last valid deadline while the file is being rewritten
        return budget["deadline"]
    if time.time() >= deadline():
        raise SystemExit("the cloud budget deadline has passed (or is too close); not starting")
    c = Config.from_dict(json.loads(args.config.read_text())) if args.config else Config()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train = AnnotatedDataset(args.train, c, voiced_vowels=not args.recorded_voicing, cache=not args.stream)
    validation = AnnotatedDataset(args.validation, c, voiced_vowels=not args.recorded_voicing)
    check_leakage(train, validation)
    if train.annotation != validation.annotation:
        raise ValueError("train and validation manifests come from different annotation runs")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("output directory is not empty; use --resume or a new directory")
    pool = balanced_indices(train, args.subset) if args.subset else list(range(len(train)))
    validation_indices = balanced_indices(validation, args.validation_size)
    validation_frames = args.validation_frames or args.batch_frames
    if not args.no_preload and not args.stream:
        train.preload(pool)
        validation.preload(validation_indices)
    system, critics = TrainingSystem(c).to(args.device), Discriminators(mel=args.mel_critic).to(args.device)
    count = sum(p.numel() for p in system.model.parameters())
    if count > c.max_parameters:
        raise ValueError(f"inference parameter budget exceeded: {count:,} > {c.max_parameters:,} (config max_parameters)")
    adaptation = None
    trainable = list(system.parameters())
    if args.adapt_from:
        source = torch.load(args.adapt_from, map_location="cpu", weights_only=True)
        if source.get("format_version") != CHECKPOINT_FORMAT:
            raise ValueError("adaptation source is not a format-4 checkpoint")
        source_config = Config.from_dict(source["config"])
        mismatch = {k: (v, getattr(c, k)) for k, v in source_config.to_dict().items() if k != "speakers" and getattr(c, k) != v}
        if source_config.language != c.language:
            mismatch["language"] = (source_config.language, c.language)
        if mismatch:
            raise ValueError(f"adaptation needs the same language profile, DSP and architecture; differs: {mismatch}")
        source_frontend = source["frontend"]["fingerprint"] if isinstance(source["frontend"], dict) else source["frontend"]
        if source_frontend != frontend_fingerprint(c.profile):
            raise ValueError("adaptation source was trained with a different front end / language profile")
        adaptation = {"source": str(args.adapt_from), "source_sha256": file_hash(args.adapt_from),
                      "source_step": source["step"], "source_speakers": list(source_config.speakers), "mode": args.adapt_mode,
                      "init_voice": args.init_voice or source_config.speakers[0]}
        adaptation["tensors"] = adaptation_init(system, source, adaptation["init_voice"])
        del source
    if args.resume:
        peek = torch.load(args.resume, map_location="cpu", weights_only=True)["recipe"]
        adaptation = peek.get("adaptation")
        del peek
    if adaptation is not None:
        trainable = adaptation_parameters(system, adaptation["mode"])
        adaptation["trainable_parameters"] = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=0.01)
    critic_optimizer = torch.optim.AdamW(critics.parameters(), lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=0.0)
    sampler = BucketSampler([train.lengths[i] for i in pool], args.batch_frames, args.max_batch, args.seed)
    fingerprints = [train.fingerprint, validation.fingerprint]
    recipe = {key: getattr(args, key) for key in ("batch_frames", "max_batch", "learning_rate", "warmup", "lr_half_life", "grad_clip",
                                                   "gan_start", "gan_ratio", "gan_ramp", "feature_matching", "gan_scale_max",
                                                   "crop_samples", "crops", "teacher_hold", "teacher_ramp", "teacher_final",
                                                   "subset", "validation_size", "seed")}
    recipe["voiced_vowel_targets"] = not args.recorded_voicing
    if args.mel_critic:
        recipe["mel_critic"] = True
    if args.pitch_delta:
        recipe["pitch_delta"] = args.pitch_delta
    if args.texture:
        recipe["texture"] = args.texture
    recipe["weights"] = {**WEIGHTS, **{key: float(value) for key, value in args.weights.items()}}
    recipe["branched_from"] = None
    recipe["adaptation"] = adaptation
    recipe["validation_frames"] = validation_frames
    recipe.update(schedule=args.lr_schedule, lr_floor=args.lr_floor, ema_decay=args.ema_decay)
    # The cosine schedule ends at this run's final step, so the target is part of the recipe.
    recipe["total_steps"] = args.steps if args.lr_schedule == "cosine" else None
    step = 0

    def compatible(state, what, same_data=True, branch=False):
        """Returns the configuration fields that differ (only possible when branching)."""
        if state.get("format_version") != CHECKPOINT_FORMAT:
            raise ValueError(f"checkpoint predates phoneme tokens and supervised durations (format < 4); it cannot be {what}")
        if same_data and (state["frontend"] != frontend_fingerprint(c.profile) or state["annotation"] != train.annotation):
            raise ValueError("checkpoint was trained with a different front end or annotation run "
                             "(branch with --init-from and --new-data to fine-tune on new annotations)")
        if state["config"] == c.to_dict():
            return {}
        if not branch:
            raise ValueError("checkpoint was trained with a different model configuration (branch with --init-from)")
        return c.branch_differences(Config.from_dict(state["config"]))
    if c.speaker_embedding_dim:
        # Fingerprint voices: fill each known voice's frozen fingerprint (both voice tables) by name.
        if not args.speaker_fingerprints:
            parser.error("speaker_embedding_dim > 0 needs --speaker-fingerprints")
        table = torch.load(args.speaker_fingerprints, map_location="cpu", weights_only=False)["fingerprints"]
        missing = [v for v in c.speakers if v not in table]
        if missing:
            # Voices without a fingerprint (kept only for branch compatibility) get zeros; they must not be trained on.
            print(json.dumps({"fingerprints_missing": len(missing), "examples": missing[:5]}), flush=True)
        dim = c.speaker_embedding_dim
        rows = torch.stack([table[v].float() if v in table else torch.zeros(dim) for v in c.speakers]).to(args.device)
        for module in (system.model.voices, system.model.frame_voices):
            module.fingerprints.copy_(rows)
    ema = EMA(system.model, args.ema_decay) if args.ema_decay is not None else None
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=True)
        compatible(state, "resumed")
        recipe["branched_from"] = state["recipe"].get("branched_from")
        if state["datasets"] != fingerprints or {**RECIPE_DEFAULTS, **state["recipe"]} != {**RECIPE_DEFAULTS, **recipe}:
            raise ValueError("resume recipe or dataset mismatch (use --init-from to branch a new run with a changed recipe)")
        system.load_state_dict(state["system"])
        critics.load_state_dict(state["critics"])
        optimizer.load_state_dict(state["optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        sampler.load_state(state["sampler"])
        if ema is not None:
            ema.load_state_dict(state["ema"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if str(args.device).startswith("cuda") and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        step = state["step"]
    elif args.init_from:
        state = torch.load(args.init_from, map_location=args.device, weights_only=True)
        # The symbol inventory is fixed (vocabulary size is part of the configuration),
        # so a checkpoint can be fine-tuned on another annotation run or front end.
        differences = compatible(state, "branched", same_data=not args.new_data, branch=True)
        step = state["step"]
        loaded = load_compatible(system, state["system"])
        loaded["config_changes"] = sorted(differences)
        if args.identity_new_blocks:
            # Pre-norm residual blocks with zero output projections pass their input through unchanged, so a
            # deeper branch starts exactly at the source model's function.
            with torch.no_grad():
                parameters = dict(system.named_parameters())
                zeroed = [name for name in loaded["new"] if name.endswith(("attention.out.weight", "attention.out.bias", "project.weight", "project.bias"))]
                for name in zeroed:
                    parameters[name].zero_()
            loaded["identity_blocks"] = zeroed
        if args.new_voice_init:
            # Appended voices start from a similar existing voice instead of a random row.
            with torch.no_grad():
                tables = dict(system.named_parameters())
                for pair in args.new_voice_init:
                    new, source = pair.split("=", 1)
                    for name in SPEAKER_TABLES:
                        if name in tables:
                            tables[name][c.speakers.index(new)] = tables[name][c.speakers.index(source)]
            loaded["new_voice_init"] = list(args.new_voice_init)
        # Critics that have already been trained are taken over, otherwise a branch
        # in the adversarial phase would hand full-strength gradients from untrained
        # critics to the generator. Optimizer moments are kept when no tensor changed.
        loaded["critics"] = step > state["recipe"]["gan_start"]
        if loaded["critics"]:
            # A critic added in the branch (e.g. --mel-critic) starts fresh; the others keep their training.
            missing, unexpected = critics.load_state_dict(state["critics"], strict=False)
            if unexpected:
                raise ValueError(f"checkpoint critics not in this run: {unexpected[:3]}")
            if missing:
                loaded["new_critics"] = sorted({key.split(".")[1] for key in missing})
            else:
                critic_optimizer.load_state_dict(state["critic_optimizer"])
        loaded["optimizer"] = not loaded["grown"] and not loaded["new"] and not loaded["dropped"]
        if loaded["optimizer"]:
            optimizer.load_state_dict(state["optimizer"])
        recipe["branched_from"] = {"step": step, "recipe": state["recipe"], "datasets": state["datasets"], "loaded": loaded}
        if args.new_data:
            recipe["branched_from"].update(frontend=state["frontend"], annotation=state["annotation"])
        if ema is not None:
            # A source EMA is taken over when every tensor was copied exactly; otherwise
            # the average starts from the branched weights.
            if state.get("ema") is not None and not loaded["grown"] and not loaded["new"] and not loaded["dropped"]:
                ema.load_state_dict(state["ema"])
                ema.decay = args.ema_decay
            else:
                ema = EMA(system.model, args.ema_decay)
    if step >= args.steps:
        raise ValueError("checkpoint already reached requested total steps")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(c.to_dict(), indent=2) + "\n")
    (args.output / "recipe.json").write_text(json.dumps({**recipe, "annotation": train.annotation, "frontend": frontend_fingerprint(c.profile),
                                                         "train_utterances": len(pool), "datasets": fingerprints}, indent=2) + "\n")
    print(json.dumps({"inference_parameters": count, "training_only_parameters": sum(p.numel() for p in system.mel_head.parameters())
                      + sum(p.numel() for p in critics.parameters()), "train_utterances": len(pool), "start_step": step,
                      "branched": bool(args.init_from), "resumed": bool(args.resume)}), flush=True)
    start = time.monotonic()
    system.train()

    def snapshot():
        return {"format_version": CHECKPOINT_FORMAT, "config": c.to_dict(), "frontend": frontend_fingerprint(c.profile), "annotation": train.annotation,
                "step": step, "datasets": fingerprints, "recipe": recipe, "system": system.state_dict(), "critics": critics.state_dict(),
                "optimizer": optimizer.state_dict(), "critic_optimizer": critic_optimizer.state_dict(), "sampler": sampler.state(),
                "ema": ema.state_dict() if ema is not None else None,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if str(args.device).startswith("cuda") else None}
    if step == args.gan_start and not args.resume:
        atomic_save(snapshot(), args.output / f"step-{step:08d}.pt")
    window, window_steps, window_start = defaultdict(float), 0, time.monotonic()
    throttle = Throttle(args.throttle_file)
    reader = ThreadPoolExecutor(4) if args.stream else None
    ahead = None            # (indices, future) of the batch the sampler will hand out next

    def load(indices):
        return [train[pool[i]] for i in indices]
    while step < args.steps:
        indices = sampler.next()
        items = ahead[1].result() if ahead is not None and ahead[0] == indices else load(indices)
        upcoming = sampler.peek() if reader is not None else None
        ahead = (upcoming, reader.submit(load, upcoming)) if upcoming is not None else None
        batch = collate(items, args.device)
        metrics = optimize_step(system, critics, optimizer, critic_optimizer, batch, step, recipe, trainable)
        if ema is not None:
            ema.update(system.model)
        step += 1
        for key, value in metrics.items():
            window[key] += value
        window_steps += 1
        elapsed = time.monotonic() - start
        should_stop = (elapsed >= args.max_seconds or step == args.steps or time.time() >= deadline()
                       or (args.stop_file is not None and args.stop_file.exists()))
        retain = step == args.gan_start or step % args.keep_every == 0
        save = step % args.save_every == 0 or should_stop or retain
        if step % args.log_every == 0 or save:
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.monotonic() - window_start
            record = {key: (value if key in ("frames", "utterances") else value / window_steps) for key, value in window.items()}
            record.update(step=step, elapsed_seconds=elapsed, steps_per_second=window_steps / max(seconds, 1e-9),
                          audio_seconds_per_second=window["frames"] * c.hop / c.sample_rate / max(seconds, 1e-9),
                          learning_rate=optimizer.param_groups[0]["lr"])
            if str(args.device).startswith("cuda"):
                record.update(peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                              peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 3))
            if save:
                validation_start = time.monotonic()
                provenance = {"adapted_from": {k: v for k, v in adaptation.items() if k != "tensors"}} if adaptation else None
                if ema is None:
                    record.update(validate(system, validation, validation_indices, args.device, recipe["weights"], frame_budget=validation_frames))
                else:
                    # Validation and the exported model use the averaged weights; training continues on the raw ones.
                    with swapped_weights(system.model, ema.weights_for(system.model)):
                        record.update(validate(system, validation, validation_indices, args.device, recipe["weights"], frame_budget=validation_frames))
                        export_model(system.model, args.output / "inference.pt", step, train.annotation,
                                     {**(provenance or {}), "ema": {"decay": ema.decay, "updates": ema.updates}})
                    record["ema_updates"] = ema.updates
                record["validation_seconds"] = round(time.monotonic() - validation_start, 2)
                state = snapshot()
                atomic_save(state, args.output / "last.pt")
                if ema is None:
                    export_model(system.model, args.output / "inference.pt", step, train.annotation, provenance)
                if retain:
                    atomic_save(state, args.output / f"step-{step:08d}.pt")
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            window, window_steps, window_start = defaultdict(float), 0, time.monotonic()
        if should_stop:
            break
        window["throttle_seconds"] += throttle()


if __name__ == "__main__":
    main()
