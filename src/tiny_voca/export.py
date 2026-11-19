"""ONNX export of free-running synthesis.

    python -m tiny_voca.export --checkpoint inference.pt --output tiny-voca.onnx

The exported graph speaks one utterance: phoneme ids and stress in, waveform
out. Text is turned into ids outside the graph (``tiny_voca.text.phonemize``).
With ``--vocoder``, the neural vocoder is part of the same graph: the
source-filter waveform is analysed into the Vocos log-mel spectrogram and the
vocoder (with its pulse-train prior, from the model's own pitch and voicing)
renders the final audio, all with convolution STFTs.

``Exportable`` computes the same audio as ``TinyVoca.synthesize`` for a single
utterance, written with operations ONNX has: the short-time Fourier transforms
of the source-filter head become real convolutions with fixed kernels, the
complex filter responses are kept as separate real and imaginary parts, and
the frame expansion uses a comparison instead of a sorted search.

Inputs
    tokens       int64 [1, N]   phoneme ids, with the edge tokens
    stress       int64 [1, N]
    voice        int64 [1]      index into ``voices`` in the metadata
    speed        float32 [1]    1.0 = as predicted, larger is faster
    pitch_shift  float32 [1]    added to log F0 (0.058 is about a semitone)
    pitch_range  float32 [1]    scales pitch movement around the mean
    emotion      int64 [1]      index into ``emotions`` (only for models with an emotion input)
    intensity    float32 [1]    0 = neutral (only with ``emotion``)
Outputs
    audio        float32 [1, samples]
    durations    int64 [1, N]   frames per token (``hop`` samples each)
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .infer import load_model, match_frontend
from .model import COMPRESSION, EMOTION_SPEED, PITCH_SMOOTH_FRAMES, jitter_offset, pulse_gate
from .losses import VocosMel
from .text import frontend_fingerprint, frontend_identity
from .vocoder import load_vocoder

INPUTS = ("tokens", "stress", "voice", "speed", "pitch_shift", "pitch_range")
EMOTION_INPUTS = ("emotion", "intensity")
OUTPUTS = ("audio", "durations")


def _cepstrum_matrices(positions, fft):
    """Real and imaginary parts of rfft for unit coefficients at ``positions``."""
    buffer = torch.zeros(len(positions), fft, dtype=torch.float64)
    buffer[torch.arange(len(positions)), torch.tensor(positions)] = 1
    spectrum = torch.fft.rfft(buffer, dim=-1)
    return spectrum.real.float(), spectrum.imag.float()


def _analysis_kernels(window):
    """Convolution kernels [bins, 1, fft] giving the real and imaginary STFT."""
    fft = len(window)
    spectrum = torch.fft.rfft(torch.eye(fft, dtype=torch.float64) * window.double(), dim=-1)
    return spectrum.real.T[:, None].float().contiguous(), spectrum.imag.T[:, None].float().contiguous()


def _synthesis_kernel(window):
    """Transposed-convolution kernel [2 * bins, 1, fft]: windowed inverse rfft
    of the real parts (first half of the channels) and imaginary parts."""
    fft = len(window)
    bins = fft // 2 + 1
    unit = torch.eye(bins, dtype=torch.complex128)
    real = torch.fft.irfft(unit, n=fft, dim=-1)
    imaginary = torch.fft.irfft(unit * 1j, n=fft, dim=-1)
    return (torch.cat((real, imaginary)) * window.double())[:, None].float().contiguous()


def _runtime_kernels(window, zero):
    """The analysis (real, imaginary), synthesis and envelope kernels of ``_analysis_kernels`` /
    ``_synthesis_kernel``, computed inside the graph from the window instead of stored: the tables are
    ~20 MB of float32 for a 24 kHz model, the windows a few kB. ``zero`` is a 0 that depends on a graph
    input, so the exporter cannot fold the computation back into constants. Angles use the exact
    integer product k * n mod fft (exact in float32 below 2**24)."""
    fft = window.shape[0]
    bins = fft // 2 + 1
    n = torch.arange(fft, dtype=torch.float32) + zero
    k = torch.arange(bins, dtype=torch.float32) + zero
    angle = torch.remainder(k[:, None] * n[None, :], float(fft)) * (2 * math.pi / fft)
    cos, sin = torch.cos(angle), torch.sin(angle)
    real, imaginary = (cos * window)[:, None], (-sin * window)[:, None]
    scale = torch.cat((torch.ones(1), torch.full((bins - 2,), 2.0), torch.ones(1))) / fft + zero
    synthesis = (torch.cat((scale[:, None] * cos, -scale[:, None] * sin)) * window)[:, None]
    return real, imaginary, synthesis


class Exportable(nn.Module):
    """Single-utterance synthesis in ONNX-friendly operations."""

    def __init__(self, model, external_noise=False, vocoder=None):
        super().__init__()
        self.model = model
        self.external_noise = external_noise
        self.emotional = model.emotions is not None
        if self.emotional:
            self.register_buffer("emotion_speed", torch.tensor([EMOTION_SPEED.get(e, 1.0) for e in model.config.emotions]), persistent=False)
        self.neural = vocoder
        if vocoder is not None:
            vc = vocoder.config
            if vc.sample_rate != model.config.sample_rate:
                raise ValueError("the vocoder and the voice model use different sample rates")
            mel = VocosMel(vc.sample_rate, vc.fft, vc.hop, vc.bands)
            self.register_buffer("mel_filters", mel.filters, persistent=False)
            self.register_buffer("neural_window", vocoder.window.float().clone(), persistent=False)
            self.register_buffer("neural_envelope", (vocoder.window.double() ** 2).float()[None, None].contiguous(), persistent=False)
            if vc.prior and vc.prior_high_hz > 0:
                # Per-voice high-band pulse share, indexed by the ``voice`` input.
                from .vocoder import pulse_position
                gains = [vc.high_gain(name) for name in model.config.speakers]
                self.register_buffer("neural_voice_gain", torch.tensor(gains, dtype=torch.float32), persistent=False)
                self.register_buffer("neural_pulse_position", pulse_position(vc), persistent=False)
        c = model.config
        v = model.vocoder
        causal = list(range(c.cep_causal))
        anti = [c.synth_fft - 1 - i for i in range(c.cep_anti)]
        for name, (real, imaginary) in (("harmonic", _cepstrum_matrices(causal + anti, c.synth_fft)),
                                        ("noise", _cepstrum_matrices(list(range(c.noise_cep)), c.noise_fft))):
            self.register_buffer(f"{name}_real", real, persistent=False)
            self.register_buffer(f"{name}_imaginary", imaginary, persistent=False)
        windows = [("synth", v.synth_window), ("noise", v.noise_window)] + ([("head", v.head.window)] if v.head is not None else [])
        for name, window in windows:
            self.register_buffer(f"{name}_window", window.float().clone(), persistent=False)
            self.register_buffer(f"{name}_envelope", (window.double() ** 2).float()[None, None].contiguous(), persistent=False)
        self.register_buffer("ramp", torch.arange(c.hop, dtype=torch.float32) / c.hop, persistent=False)
        self.phase_steps = 1 << 24
        self._zero, self._kernels = None, {}

    def _kernel(self, name, part):
        """STFT kernels of window ``name`` (0 analysis real, 1 analysis imaginary, 2 synthesis), built once
        per forward pass from the window (see ``_runtime_kernels``)."""
        if name not in self._kernels:
            zero = self._zero if self._zero is not None else torch.zeros(())
            self._kernels[name] = _runtime_kernels(getattr(self, f"{name}_window"), zero)
        return self._kernels[name][part]

    def _filter(self, source, real, imaginary, name, fft, hop):
        """Multiply the STFT of ``source`` [1, samples] by a response given as
        real and imaginary parts [1, frames, bins], and resynthesize."""
        samples = source.shape[1]
        padded = F.pad(source[:, None], (fft // 2, fft))
        s_real = F.conv1d(padded, self._kernel(name, 0), stride=hop)
        s_imaginary = F.conv1d(padded, self._kernel(name, 1), stride=hop)
        real, imaginary = real.transpose(1, 2), imaginary.transpose(1, 2)
        product = torch.cat((s_real * real - s_imaginary * imaginary, s_real * imaginary + s_imaginary * real), dim=1)
        wave = F.conv_transpose1d(product, self._kernel(name, 2), stride=hop)
        envelope = F.conv_transpose1d(torch.ones_like(s_real[:, :1]), getattr(self, f"{name}_envelope"), stride=hop)
        return (wave / envelope)[:, 0, fft // 2:fft // 2 + samples]

    @staticmethod
    def _response(cepstrum, real, imaginary):
        magnitude = torch.exp((cepstrum @ real).clamp(-14, 6))
        angle = cepstrum @ imaginary
        return magnitude * torch.cos(angle), magnitude * torch.sin(angle)

    def _pulses(self, f0, log_offset=None, limit=None):
        """``pulse_train`` for one utterance, without 64-bit floats (the web
        runtime has none). The phase is summed inside each frame in float32,
        where it stays below a few cycles, and across frames as 24-bit integer
        fractions of a cycle, which wrap exactly however long the utterance is."""
        c = self.model.config
        knots = torch.log(f0.clamp_min(1.0))
        knots = torch.cat((knots, knots[:, -1:]), dim=1)
        low, high = knots[:, :-1, None], knots[:, 1:, None]
        log_frequency = low * (1 - self.ramp) + high * self.ramp
        if log_offset is not None:
            log_frequency = log_frequency + log_offset.reshape(1, -1, c.hop)
        frequency = torch.exp(log_frequency)
        within = torch.cumsum(frequency / c.sample_rate, dim=2)
        whole_frame = within[:, :, -1]
        steps = torch.round((whole_frame - torch.floor(whole_frame)) * self.phase_steps).long()
        start = torch.remainder(torch.cumsum(steps, dim=1) - steps, self.phase_steps).to(within.dtype) / self.phase_steps
        cycles = within + start[:, :, None]
        theta = (2 * math.pi * (cycles - torch.floor(cycles))).reshape(1, -1)
        frequency = frequency.reshape(1, -1)
        harmonics = ((c.harmonic_limit if limit is None else limit) * c.sample_rate / frequency).clamp_min(1.0)
        whole = torch.floor(harmonics)
        denominator = 2 * torch.sin(theta / 2)
        tiny = denominator.abs() < 1e-6
        core = torch.where(tiny, whole, torch.sin((whole + 0.5) * theta) / torch.where(tiny, torch.ones_like(denominator), denominator) - 0.5)
        wave = core + (harmonics - whole) * torch.cos((whole + 1) * theta)
        return wave * torch.sqrt(2.0 / harmonics)

    def _vocoder(self, hidden, f0, total, noise, voiced=None):
        c = self.model.config
        v = self.model.vocoder
        last = total - 1
        # Harmonic branch.
        h = v.harmonic(hidden)
        split = c.cep_causal + c.cep_anti
        steps = v.harmonic_steps
        sub = torch.minimum(torch.arange(steps * total + 3, device=hidden.device), steps * total - 1)
        here = torch.div(sub, steps, rounding_mode="floor")
        following = torch.minimum(torch.div(sub + 1, steps, rounding_mode="floor"), last)
        envelope = 0.5 * (h[..., :split].index_select(1, here) + h[..., :split].index_select(1, following))
        envelope = envelope * torch.cat((v.causal_lifter, v.anti_lifter))
        gain = h[..., split:].flatten(1).index_select(1, sub)[..., None]
        envelope = torch.cat((envelope[..., :1] + gain, envelope[..., 1:]), dim=-1)
        harmonic_real, harmonic_imaginary = self._response(envelope, self.harmonic_real, self.harmonic_imaginary)
        # Noise branch.
        n = v.noise(hidden)
        steps = v.noise_steps
        sub = torch.minimum(torch.arange(steps * total + 3, device=hidden.device), steps * total - 1)
        shape = n[..., :c.noise_cep].index_select(1, torch.div(sub, steps, rounding_mode="floor")) * v.noise_lifter
        gain = n[..., c.noise_cep:].flatten(1).index_select(1, sub)[..., None]
        shape = torch.cat((shape[..., :1] + gain, shape[..., 1:]), dim=-1)
        noise_real, noise_imaginary = self._response(shape, self.noise_real, self.noise_imaginary)
        if noise is None:
            noise = torch.randn_like(f0[:, :, None].expand(-1, -1, c.hop).reshape(1, -1))
        else:
            noise = noise.index_select(1, torch.arange(c.hop * total, device=hidden.device))
        offset = jitter_offset(noise, c.excitation_jitter) if c.excitation_jitter > 0 else None
        pulses = self._pulses(f0, offset)
        if c.unvoiced_pulse_gain < 1 and voiced is not None:
            pulses = pulses * pulse_gate(voiced, c.hop, c.unvoiced_pulse_gain)
        wave = (self._filter(pulses, harmonic_real, harmonic_imaginary, "synth", c.synth_fft, c.synth_hop)
                + self._filter(noise, noise_real, noise_imaginary, "noise", c.noise_fft, c.noise_hop))
        wave = wave if v.postfilter is None else self._postfilter(wave, hidden, total)
        return wave if v.head is None else self._head(wave, hidden, total)

    def _stft(self, wave):
        """Centred, reflection-padded STFT (as torch.stft) -> real, imaginary [1, bins, samples // hop + 1]."""
        fft, hop = self.neural.config.fft, self.neural.config.hop
        padded = F.pad(wave[:, None], (fft // 2, fft // 2), mode="reflect")
        return (F.conv1d(padded, self._kernel("neural", 0), stride=hop),
                F.conv1d(padded, self._kernel("neural", 1), stride=hop))

    def _neural(self, wave, f0, voiced, total, voice):
        """Source-filter waveform -> Vocos log-mel -> neural vocoder (``Vocoder.forward``)."""
        v = self.neural
        vc = v.config
        real, imaginary = self._stft(wave)
        mel = torch.log((self.mel_filters @ torch.sqrt(real * real + imaginary * imaginary)).clamp_min(1e-7))
        log_prior = None
        if vc.prior:
            samples = wave.shape[1]
            gate = F.interpolate(voiced[:, None], size=samples, mode="linear", align_corners=False)[:, 0]
            pulses = self._pulses(f0.clamp(40, 1000), limit=0.47) * gate
            noise = vc.prior_noise * torch.randn_like(wave)
            if vc.prior_high_hz > 0:
                pulse_real, pulse_imaginary = self._stft(pulses)
                noise_real, noise_imaginary = self._stft(noise)
                gain = self.neural_voice_gain[voice][:, None, None]
                weight = 1 + (gain - 1) * self.neural_pulse_position[None, :, None]
                prior_real, prior_imaginary = pulse_real * weight + noise_real, pulse_imaginary * weight + noise_imaginary
            else:
                prior_real, prior_imaginary = self._stft(pulses + noise)
            prior_magnitude = torch.sqrt(prior_real * prior_real + prior_imaginary * prior_imaginary)
            log_prior = torch.log(prior_magnitude + 1e-5)
        log_magnitude, phase = v.spectrum_from_log_prior(mel, log_prior)
        magnitude = torch.exp(log_magnitude).clamp(max=1e2)
        cos, sin = torch.cos(phase), torch.sin(phase)
        if vc.prior:
            # Rotate the prior's unit phasor by the predicted correction (no atan2 needed).
            zero = prior_magnitude == 0
            unit_real = torch.where(zero, torch.ones_like(prior_real), prior_real / torch.where(zero, torch.ones_like(prior_magnitude), prior_magnitude))
            unit_imaginary = torch.where(zero, torch.zeros_like(prior_imaginary), prior_imaginary / torch.where(zero, torch.ones_like(prior_magnitude), prior_magnitude))
            cos, sin = unit_real * cos - unit_imaginary * sin, unit_real * sin + unit_imaginary * cos
        spectrum = torch.cat((magnitude * cos, magnitude * sin), dim=1)
        out = F.conv_transpose1d(spectrum, self._kernel("neural", 2), stride=vc.hop)
        envelope = F.conv_transpose1d(torch.ones_like(magnitude[:, :1]), self.neural_envelope, stride=vc.hop)
        fft = vc.fft
        return (out / envelope)[:, 0, fft // 2:fft // 2 + total * vc.hop]

    def _head(self, wave, hidden, total):
        """``SpectralHead`` for one utterance with convolution STFTs (same frames
        as torch.stft/istft with center=True and half a window of trailing zeros)."""
        h = self.model.vocoder.head
        fft, hop, samples = h.fft, h.hop, wave.shape[1]
        padded = F.pad(wave[:, None], (fft // 2, fft))
        real = F.conv1d(padded, self._kernel("head", 0), stride=hop)                # [1, bins, frames]
        imaginary = F.conv1d(padded, self._kernel("head", 1), stride=hop)
        index = torch.minimum(torch.arange(real.shape[-1], device=wave.device), total - 1)
        condition = hidden.index_select(1, index).transpose(1, 2)
        real, imaginary = h.correct(real, imaginary, condition, 1.0)
        out = F.conv_transpose1d(torch.cat((real, imaginary), dim=1), self._kernel("head", 2), stride=hop)
        envelope = F.conv_transpose1d(torch.ones_like(real[:, :1]), self.head_envelope, stride=hop)
        return (out / envelope)[:, 0, fft // 2:fft // 2 + samples]

    def _postfilter(self, wave, hidden, total):
        """``SpectralPostFilter`` for one utterance, with convolution STFTs in
        place of complex tensors (same frames as torch.stft/istft, center=True)."""
        p = self.model.vocoder.postfilter
        fft, hop, samples = p.fft, p.hop, wave.shape[1]
        padded = F.pad(wave[:, None], (fft // 2, fft))     # centring plus half a window of trailing zeros
        real = F.conv1d(padded, self._kernel("synth", 0), stride=hop)              # [1, bins, frames]
        imaginary = F.conv1d(padded, self._kernel("synth", 1), stride=hop)
        magnitude = torch.sqrt(real * real + imaginary * imaginary + 1e-10)
        scale = (magnitude + 1e-4) ** (COMPRESSION - 1)
        index = torch.minimum(torch.div(torch.arange(real.shape[-1], device=wave.device), p.steps, rounding_mode="floor"),
                              total - 1)
        condition = p.condition(hidden).index_select(1, index).transpose(1, 2)     # [1, bins, frames]
        x = p.input(torch.stack((real * scale, imaginary * scale, torch.log(magnitude + 1e-4), condition), dim=1))
        for block in p.blocks:
            x = block(x, 1.0)
        correction = p.output(x)
        level = (magnitude + 1e-4) ** (1 - COMPRESSION)
        spectrum = torch.cat((real + correction[:, 0] * level, imaginary + correction[:, 1] * level), dim=1)
        out = F.conv_transpose1d(spectrum, self._kernel("synth", 2), stride=hop)
        envelope = F.conv_transpose1d(torch.ones_like(real[:, :1]), self.synth_envelope, stride=hop)
        return (out / envelope)[:, 0, fft // 2:fft // 2 + samples]

    def forward(self, tokens, stress, voice, speed, pitch_shift, pitch_range, *extra):
        self._zero, self._kernels = (speed * 0).sum(), {}
        m = self.model
        c = m.config
        if self.emotional:
            emotion, intensity, *extra = extra
            # Emotion tempo (model.EMOTION_SPEED), linear in intensity, as in TinyVoca.synthesize.
            speed = speed * (1 + (self.emotion_speed[emotion] - 1) * intensity)
        noise = extra[0] if extra else None
        token_mask = torch.ones_like(tokens, dtype=torch.bool)
        x = m.tokens(tokens) + m.stress(stress) + m.voices(voice)[:, None]
        if self.emotional:
            x = x + (m.emotions(emotion) * intensity[:, None])[:, None]
        hidden = m.encoder(x, token_mask)
        # Durations.
        output = m.duration(hidden, token_mask)
        frames = torch.round((torch.exp(output[..., 0].clamp(0, math.log1p(400))) - 1) / speed).long().clamp_min(1)
        present = m.phone_tokens[tokens] | (output[..., 1] > 0)
        durations = torch.where(present, frames, torch.zeros_like(frames))
        # Repeat token states by duration.
        ends = durations.cumsum(1)
        total = ends[0, -1]
        frame = torch.arange(total, device=tokens.device)[None]
        index = (ends[:, None, :] <= frame[:, :, None]).sum(-1)
        owned = durations.gather(1, index).clamp_min(1)
        offset = frame - (ends.gather(1, index) - durations.gather(1, index))
        relative = (offset.to(hidden.dtype) + 0.5) / owned
        features = torch.stack((relative, 1 - relative, torch.log1p(owned.to(hidden.dtype)) / 4), dim=-1)
        expanded = hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))
        expanded = expanded + m.frame_position(features) + m.frame_voices(voice)[:, None]
        if self.emotional:
            expanded = expanded + (m.frame_emotions(emotion) * intensity[:, None])[:, None]
        frame_mask = torch.ones_like(frame, dtype=torch.bool)
        # Pitch, voicing, energy.
        prosody_hidden = m.prosody_stack(expanded, frame_mask)
        prosody = m.prosody_head(prosody_hidden)
        vowel = m.vowel_tokens[tokens].gather(1, index)
        voicing = torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])
        voiced = (voicing > 0).to(prosody.dtype)
        log_f0 = m.smooth_pitch(prosody[..., 0], voicing > 0, frame_mask) if PITCH_SMOOTH_FRAMES > 1 else prosody[..., 0]
        mean = (log_f0 * voiced).sum(1, keepdim=True) / voiced.sum(1, keepdim=True).clamp_min(1)
        pitch = mean + (log_f0 - mean) * pitch_range
        condition = torch.stack((pitch.clamp(-2.0, 1.5), voiced, prosody[..., 2].clamp(-4.5, 3.0)), dim=-1)
        acoustic = m.acoustic_stack(prosody_hidden + m.prosody_projection(condition), frame_mask)
        f0 = (c.pitch_reference_hz * torch.exp(pitch + pitch_shift)).clamp(c.f0_min, c.f0_max)
        wave = self._vocoder(acoustic, f0, total, noise, voiced)
        return (wave if self.neural is None else self._neural(wave, f0, voiced, total, voice)), durations


def example_inputs(length=24, vocab_size=76):
    tokens = torch.randint(5, vocab_size, (1, length))
    one = lambda value: torch.tensor([value], dtype=torch.float32)
    return tokens, torch.zeros_like(tokens), torch.tensor([0]), one(1.0), one(0.0), one(1.0)


def export(model, path, external_noise=False, opset=17, vocoder=None):
    wrapper = Exportable(model, external_noise, vocoder).eval()
    inputs = example_inputs(vocab_size=model.config.vocab_size)
    names = list(INPUTS)
    if wrapper.emotional:
        inputs = (*inputs, torch.tensor([0]), torch.tensor([0.0]))
        names += list(EMOTION_INPUTS)
    axes = {"tokens": {1: "tokens"}, "stress": {1: "tokens"}, "audio": {1: "samples"}, "durations": {1: "tokens"}}
    if external_noise:
        inputs = (*inputs, torch.randn(1, model.config.hop * model.config.max_frames))
        names.append("noise")
        axes["noise"] = {1: "noise_samples"}
    with torch.no_grad():
        torch.onnx.export(wrapper, inputs, str(path), input_names=names, output_names=list(OUTPUTS), dynamic_axes=axes,
                          opset_version=opset, do_constant_folding=True, dynamo=False)
    return path


def metadata(model, checkpoint, vocoder_checkpoint=None):
    c = model.config
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    profile = c.profile
    value = {"sample_rate": c.sample_rate, "hop": c.hop, "voices": list(c.speakers), "symbols": list(profile.symbols),
             "max_tokens": c.max_tokens, "max_frames": c.max_frames, "step": state.get("step"),
             "frontend": frontend_identity(profile).get("phonemizer", "espeak"), "frontend_fingerprint": frontend_fingerprint(profile),
             "inputs": list(INPUTS) + (list(EMOTION_INPUTS) if c.emotions else []), "outputs": list(OUTPUTS)}
    if c.emotions:
        value["emotions"] = list(c.emotions)
    if vocoder_checkpoint is not None:
        state = torch.load(vocoder_checkpoint, map_location="cpu", weights_only=True)
        value["vocoder_step"] = state.get("step")
        value["vocoder_voice_high_gain"] = state["config"].get("voice_high_gain", [])
    if profile.language != "en":
        # The full front-end rules (inventory, eSpeak canonicalisation, geminates, abbreviations),
        # so a non-Python front end can reproduce the tokens from the same table.
        value.update(language=profile.language, locale=profile.locale, profile_version=profile.version,
                     profile_digest=profile.digest, espeak_voice=profile.espeak_voice,
                     frontend_identity=frontend_identity(profile))
    return value


def main():
    parser = argparse.ArgumentParser(description="Export a tiny-voca inference checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-noise", action="store_true", help="Take the noise signal as an input (for exact comparisons)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--vocoder", type=Path, help="Neural vocoder checkpoint to include in the graph")
    args = parser.parse_args()
    match_frontend(args.checkpoint)
    model = load_model(args.checkpoint, "cpu")
    vocoder = load_vocoder(args.vocoder) if args.vocoder else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export(model, args.output, args.external_noise, args.opset, vocoder)
    args.output.with_suffix(".json").write_text(json.dumps(metadata(model, args.checkpoint, args.vocoder), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB) and {args.output.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
