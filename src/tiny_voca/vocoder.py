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

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{"prior": False, "prior_noise": 0.03, **value})


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

    def analyse_prior(self, f0, voiced, noise=None):
        """Pitch [B, frames] in Hz and voicing in [0, 1] -> complex STFT [B, bins, frames + 1]
        of a pulse train (voiced frames) plus a fixed noise floor."""
        from .model import pulse_train
        c = self.config
        samples = f0.shape[1] * c.hop
        pulses = pulse_train(f0.clamp(40, 1000), c.sample_rate, c.hop)
        gate = F.interpolate(voiced.float()[:, None], size=samples, mode="linear", align_corners=False)[:, 0]
        if noise is None:
            noise = torch.randn(f0.shape[0], samples, device=f0.device)
        source = pulses * gate + c.prior_noise * noise
        return torch.stft(source, c.fft, c.hop, window=self.window, center=True, return_complex=True)

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

    def forward(self, mel, f0=None, voiced=None, noise=None):
        """mel [B, bands, frames + 1] (and, with a prior, f0/voiced [B, frames]) -> wave [B, frames * hop]."""
        prior = self.analyse_prior(f0, voiced, noise) if self.config.prior else None
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
