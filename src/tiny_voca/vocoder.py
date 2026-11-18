"""Small neural vocoder for tiny-voca: log-mel frames -> 24 kHz waveform.

A Vocos-style network (Siuzdak, 2023): ConvNeXt blocks over mel frames and a
head that predicts the log magnitude and phase of every STFT bin, inverted with
one inverse STFT. All layers run at the frame rate (94 frames/s), which keeps the
cost low on CPU. The input is the Vocos 24 kHz log-mel (``losses.VocosMel``, 100
bands, hop 256), so the pretrained Vocos can serve as a reference on identical
features.

With ``prior=True`` (after Wavehax, Yoneyama et al. 2024, and the neural
source-filter vocoders) the vocoder also receives the frame pitch: it builds a
band-limited pulse train for voiced frames plus a little noise, and the phase of
that signal's STFT is the starting point of the output phase. The network then
predicts every bin's log magnitude (the prior's log magnitude is an input) and only
a phase *correction*. Harmonics advance consistently from frame to frame by
construction, which a small network otherwise has to invent - the usual source of
the metallic sound of small iSTFT vocoders.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VocoderConfig:
    sample_rate: int = 24000
    bands: int = 100
    fft: int = 1024
    hop: int = 256
    channels: int = 192
    intermediate: int = 576
    layers: int = 8
    prior: bool = False
    prior_noise: float = 0.03
    # Pulse share of the prior above prior_high_hz, fading in over one octave (0 Hz = off).
    # Real speech is less pulse-locked at high frequencies than the pulse train.
    prior_high_hz: float = 0.0
    prior_high_gain: float = 1.0
    # Optional per-voice override of prior_high_gain: ((voice name, gain), ...). Voices
    # differ in how pulse-locked their high band is (IT1 is breathier than EF3).
    voice_high_gain: tuple = ()

    def to_dict(self):
        value = asdict(self)
        value["voice_high_gain"] = [list(pair) for pair in self.voice_high_gain]
        return value

    @classmethod
    def from_dict(cls, value):
        value = {"prior": False, "prior_noise": 0.03, "prior_high_hz": 0.0, "prior_high_gain": 1.0, **value}
        value["voice_high_gain"] = tuple((str(name), float(gain)) for name, gain in value.get("voice_high_gain", ()))
        return cls(**value)

    def high_gain(self, voice=None):
        """The high-band pulse share for a voice name (the default when it has no override)."""
        return dict(self.voice_high_gain).get(voice, self.prior_high_gain)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, intermediate, scale):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, intermediate)
        self.project = nn.Linear(intermediate, channels)
        self.scale = nn.Parameter(torch.full((channels,), scale))

    def forward(self, x):                                   # x [B, C, frames]
        y = self.norm(self.depthwise(x).transpose(1, 2))
        y = self.project(F.gelu(self.expand(y))) * self.scale
        return x + y.transpose(1, 2)


def pulse_position(c):
    """[bins] 0 below ``prior_high_hz``, rising linearly in log frequency to 1 at twice it."""
    frequencies = torch.linspace(0, c.sample_rate / 2, c.fft // 2 + 1)
    if c.prior_high_hz <= 0:
        return torch.zeros_like(frequencies)
    return (torch.log2(frequencies.clamp_min(1.0) / c.prior_high_hz)).clamp(0, 1)


def pulse_weight(c, gain=None):
    """[bins] weight of the pulse train in the prior: 1 below ``prior_high_hz``, then a
    linear fade (in log frequency) to ``gain`` (default ``prior_high_gain``) at twice that frequency."""
    return 1 + ((c.prior_high_gain if gain is None else gain) - 1) * pulse_position(c)


class Vocoder(nn.Module):
    def __init__(self, config=VocoderConfig()):
        super().__init__()
        self.config = c = config
        self.embed = nn.Conv1d(c.bands, c.channels, 7, padding=3)
        self.norm = nn.LayerNorm(c.channels, eps=1e-6)
        self.blocks = nn.ModuleList(ConvNeXtBlock(c.channels, c.intermediate, 1.0 / c.layers) for _ in range(c.layers))
        self.final = nn.LayerNorm(c.channels, eps=1e-6)
        self.head = nn.Linear(c.channels, c.fft + 2)
        if c.prior:
            self.prior_in = nn.Conv1d(c.fft // 2 + 1, c.channels, 1)
            nn.init.normal_(self.prior_in.weight, std=0.01)
            nn.init.zeros_(self.prior_in.bias)
        self.register_buffer("window", torch.hann_window(c.fft), persistent=False)
        self.register_buffer("pulse_weight", pulse_weight(c), persistent=False)
        self.register_buffer("pulse_position", pulse_position(c), persistent=False)

    def analyse_prior(self, f0, voiced, noise=None, gain=None):
        """Pitch [B, frames] in Hz and voicing in [0, 1] -> complex STFT [B, bins, frames + 1]
        of a pulse train (voiced frames) plus a fixed noise floor. ``gain`` (a number, or one
        per utterance [B]) overrides the high-band pulse share."""
        from .model import pulse_train
        c = self.config
        samples = f0.shape[1] * c.hop
        pulses = pulse_train(f0.clamp(40, 1000), c.sample_rate, c.hop)
        gate = F.interpolate(voiced.float()[:, None], size=samples, mode="linear", align_corners=False)[:, 0]
        if noise is None:
            noise = torch.randn(f0.shape[0], samples, device=f0.device)
        if c.prior_high_hz <= 0:
            source = pulses * gate + c.prior_noise * noise
            return torch.stft(source, c.fft, c.hop, window=self.window, center=True, return_complex=True)
        stft = lambda x: torch.stft(x, c.fft, c.hop, window=self.window, center=True, return_complex=True)
        if gain is None:
            weight = self.pulse_weight[None, :, None]
        else:
            gain = torch.as_tensor(gain, dtype=self.pulse_position.dtype, device=self.pulse_position.device).reshape(-1, 1, 1)
            weight = 1 + (gain - 1) * self.pulse_position[None, :, None]
        return stft(pulses * gate) * weight + stft(c.prior_noise * noise)

    def spectrum(self, mel, prior=None):
        """mel [B, bands, frames] -> (log magnitude, phase or phase correction), each [B, bins, frames]."""
        return self.spectrum_from_log_prior(mel, torch.log(prior.abs() + 1e-5) if self.config.prior else None)

    def spectrum_from_log_prior(self, mel, log_prior=None):
        """``spectrum`` given the prior's log magnitude [B, bins, frames] instead of its complex STFT."""
        x = self.embed(mel)
        if self.config.prior:
            x = x + self.prior_in(log_prior)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.head(self.final(x.transpose(1, 2))).transpose(1, 2)
        bins = self.config.fft // 2 + 1
        return x[:, :bins], x[:, bins:]

    def forward(self, mel, f0=None, voiced=None, noise=None, voice=None):
        """mel [B, bands, frames + 1] (and, with a prior, f0/voiced [B, frames]) -> wave [B, frames * hop].
        ``voice`` (a name) selects a per-voice high-band pulse share, if the config has one."""
        gain = self.config.high_gain(voice) if voice is not None else None
        prior = self.analyse_prior(f0, voiced, noise, gain) if self.config.prior else None
        return self.render(*self.spectrum(mel, prior), prior=prior)

    def render(self, log_magnitude, phase, prior=None):
        """Inverse STFT of a predicted spectrum (the second half of ``forward``)."""
        if prior is not None:
            phase = torch.angle(prior) + phase
        magnitude = torch.exp(log_magnitude).clamp(max=1e2)
        spectrum = torch.polar(magnitude, phase)
        c = self.config
        return torch.istft(spectrum, c.fft, c.hop, window=self.window, center=True)


def load_vocoder(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = Vocoder(VocoderConfig.from_dict(state["config"]))
    model.load_state_dict(state["vocoder"])
    return model.to(device).eval()
