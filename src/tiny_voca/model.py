"""tiny-voca v2: feed-forward transformer with a pitch-driven source-filter head.

All weights start random. Tensor conventions: hidden [B,T,C], frame features
[B,T], waveform [B,S]. Acoustic frame i is centred on sample ``i * hop``.

    phoneme + stress + voice -> transformer encoder -> duration predictor
        -> repeat tokens by duration (supervised durations while training)
        -> frame transformer A -> pitch / voicing / energy
        -> frame transformer B -> filter parameters
        -> harmonic pulse train (from pitch) and noise, each shaped by its
           time-varying filter in the STFT domain -> 24 kHz waveform

The waveform head has no neural upsampling network. The oscillator supplies the
periodicity that a tiny decoder cannot render reliably on its own; the network
only has to predict slowly varying spectral envelopes (complex cepstra), which
is what keeps the voice from sounding buzzy or unstable at this size.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import Config

# Frames of the inference-time pitch smoothing (TinyVoca.smooth_pitch); 1 = off.
PITCH_SMOOTH_FRAMES = 5


def length_mask(lengths, size=None):
    size = int(lengths.max()) if size is None else size
    return torch.arange(size, device=lengths.device)[None, :] < lengths[:, None]


def rotary_table(size, width, device):
    inverse = 1.0 / (10000 ** (torch.arange(0, width, 2, device=device, dtype=torch.float32) / width))
    angle = torch.arange(size, device=device, dtype=torch.float32)[:, None] * inverse[None]
    return angle.cos(), angle.sin()


def rotate(x, table):
    cos, sin = table[0].to(x.dtype), table[1].to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Self-attention with rotary (relative) positions: no absolute position
    table, so utterances longer than those seen in training stay in range."""

    def __init__(self, channels, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out = nn.Linear(channels, channels)

    def forward(self, x, mask, table):
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(rotate(q, table), rotate(k, table), v, attn_mask=mask[:, None, None, :],
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, c))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.channels)
        self.attention = Attention(c.channels, c.heads, c.dropout)
        self.norm2 = nn.LayerNorm(c.channels)
        self.depthwise = nn.Conv1d(c.channels, c.channels, c.kernel, padding=c.kernel // 2, groups=c.channels)
        self.expand = nn.Linear(c.channels, c.feedforward)
        self.project = nn.Linear(c.feedforward, c.channels)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x, mask, table):
        keep = mask[..., None]
        x = (x + self.dropout(self.attention(self.norm1(x), mask, table))) * keep
        # Mask before the convolution so padded positions cannot leak back in.
        y = self.depthwise((self.norm2(x) * keep).transpose(1, 2)).transpose(1, 2)
        y = self.project(self.dropout(F.gelu(self.expand(y))))
        return (x + self.dropout(y)) * keep


class Stack(nn.Module):
    def __init__(self, c, layers):
        super().__init__()
        self.blocks = nn.ModuleList(Block(c) for _ in range(layers))
        self.norm = nn.LayerNorm(c.channels)
        self.width = c.channels // c.heads

    def forward(self, x, mask):
        table = rotary_table(x.shape[1], self.width, x.device)
        for block in self.blocks:
            x = block(x, mask, table)
        return self.norm(x) * mask[..., None]


class Predictor(nn.Module):
    def __init__(self, c, outputs):
        super().__init__()
        p = c.predictor_channels
        self.conv1 = nn.Conv1d(c.channels, p, 3, padding=1)
        self.conv2 = nn.Conv1d(p, p, 3, padding=1)
        self.norm1, self.norm2 = nn.LayerNorm(p), nn.LayerNorm(p)
        self.drop = nn.Dropout(c.dropout)
        self.out = nn.Linear(p, outputs)

    def forward(self, x, mask):
        x = self.conv1((x * mask[..., None]).transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.norm1(F.gelu(x))) * mask[..., None]
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.norm2(F.gelu(x)))
        return self.out(x) * mask[..., None]


JITTER_TAPS = 25   # about 1 ms at 24 kHz


def jitter_offset(noise, amount):
    """Log-frequency perturbation [B, samples] for the pulse train, made from the
    same white noise that drives the noise branch: smoothed over about 1 ms and
    scaled to standard deviation ``amount``. Deterministic given the noise, so
    PyTorch and ONNX synthesis stay identical."""
    kernel = torch.hann_window(JITTER_TAPS + 2, periodic=False, device=noise.device, dtype=noise.dtype)[1:-1]
    kernel = (kernel / kernel.norm())[None, None]
    return amount * F.conv1d(noise[:, None], kernel, padding=JITTER_TAPS // 2)[:, 0]


GATE_FADE = 601    # samples, about 25 ms at 24 kHz


def pulse_gate(voiced, hop, gain):
    """[B, frames] voicing in [0, 1] -> [B, frames * hop] multiplier for the pulse train:
    1 in voiced frames, ``gain`` in unvoiced ones, linearly interpolated and smoothed
    over about 25 ms so the switch is soft."""
    samples = voiced.shape[1] * hop
    gate = F.interpolate(voiced.float()[:, None], size=samples, mode="linear", align_corners=False)[:, 0]
    pad = GATE_FADE // 2
    gate = F.avg_pool1d(F.pad(gate[:, None], (pad, pad), mode="replicate"), GATE_FADE, 1)[:, 0]
    return gain + (1 - gain) * gate


def pulse_train(f0, sample_rate, hop, limit=0.47, phase=None, log_offset=None):
    """Band-limited pulse train [B, T*hop] with unit power for f0 [B, T] in Hz.

    Frame values sit at samples i*hop and are interpolated in the log domain.
    The sum of all harmonics below ``limit * sample_rate`` has the closed form
    sin((K+1/2)x) / (2 sin(x/2)) - 1/2; the top harmonic fades in and out so the
    signal stays continuous while the pitch glides. Float64 keeps the phase exact.
    ``log_offset`` [B, T*hop] is added to the interpolated log frequency (jitter).
    """
    frames = f0.shape[1]
    knots = torch.log(f0.double().clamp_min(1.0))
    knots = torch.cat((knots, knots[:, -1:]), dim=1)
    position = torch.arange(frames * hop, device=f0.device, dtype=torch.float64) / hop
    low = position.floor().long()
    weight = position - low
    log_frequency = knots[:, low] * (1 - weight) + knots[:, low + 1] * weight
    if log_offset is not None:
        log_frequency = log_frequency + log_offset.double()
    frequency = torch.exp(log_frequency)
    theta = 2 * math.pi * torch.cumsum(frequency / sample_rate, dim=1)
    if phase is not None:
        theta = theta + phase.double()[:, None]
    theta = torch.remainder(theta, 2 * math.pi)
    harmonics = (limit * sample_rate / frequency).clamp_min(1.0)
    whole = torch.floor(harmonics)
    denominator = 2 * torch.sin(theta / 2)
    tiny = denominator.abs() < 1e-7
    core = torch.where(tiny, whole, torch.sin((whole + 0.5) * theta) / torch.where(tiny, torch.ones_like(denominator), denominator) - 0.5)
    wave = core + (harmonics - whole) * torch.cos((whole + 1) * theta)
    return (wave * torch.sqrt(2.0 / harmonics)).float()


def filtered(source, response, fft, hop, window):
    """Time-varying filtering: response [B, frames, bins] multiplies the STFT."""
    spectrum = torch.stft(source, fft, hop, window=window, center=True, pad_mode="constant", return_complex=True)
    if spectrum.shape[-1] != response.shape[1]:
        raise ValueError("filter frames do not match the source")
    return torch.istft(spectrum * response.transpose(1, 2), fft, hop, window=window, center=True, length=source.shape[-1])


def cepstrum_response(causal, anticausal, fft):
    """Complex frequency response exp(FFT(cepstrum)). Causal coefficients give a
    minimum-phase filter; anticausal ones add a maximum-phase part (glottal pulse)."""
    buffer = causal.new_zeros(*causal.shape[:-1], fft)
    buffer[..., :causal.shape[-1]] = causal
    if anticausal is not None:
        buffer[..., fft - anticausal.shape[-1]:] = anticausal.flip(-1)
    spectrum = torch.fft.rfft(buffer, dim=-1)
    return torch.polar(torch.exp(spectrum.real.clamp(-14, 6)), spectrum.imag)


COMPRESSION = 0.3


class PostFilterBlock(nn.Module):
    """ConvNeXt block over a [B, C, frequency, time] map; frames past an
    utterance's end stay zero so batching never changes real audio."""

    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Conv2d(channels, 2 * channels, 1)
        self.project = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x, mask):
        y = self.norm(self.depthwise(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return (x + self.project(F.gelu(self.expand(y)))) * mask


class SpectralPostFilter(nn.Module):
    """Complex correction of the synthesized STFT (synth_fft / synth_hop), after
    Wavehax (Yoneyama et al., 2024) with the source-filter output as the harmonic
    prior. Inputs per bin: compressed real and imaginary parts, log magnitude and a
    per-frame projection of the acoustic state. The correction is scaled by the
    bin's own level and starts at zero, so a new post-filter is the identity."""

    def __init__(self, c):
        super().__init__()
        self.fft, self.hop, self.steps = c.synth_fft, c.synth_hop, c.hop // c.synth_hop
        width = c.postfilter_channels
        self.condition = nn.Linear(c.channels, c.synth_fft // 2 + 1)
        self.input = nn.Conv2d(4, width, (7, 1), padding=(3, 0))
        self.blocks = nn.ModuleList(PostFilterBlock(width) for _ in range(c.postfilter_blocks))
        self.output = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("window", torch.hann_window(c.synth_fft), persistent=False)

    def forward(self, wave, hidden, lengths):
        # Half a window of trailing zeros, and the frames that reach into it use the
        # last real frame's state: an utterance's end is then processed the same way
        # alone or padded inside a batch (frames further out are masked to zero).
        padded = F.pad(wave, (0, self.fft // 2))
        frames = padded.shape[-1] // self.hop + 1
        spectrum = torch.stft(padded, self.fft, self.hop, window=self.window, center=True, pad_mode="constant", return_complex=True)
        position = torch.arange(frames, device=wave.device)[None]
        mask = (position <= (lengths * self.steps + self.fft // (2 * self.hop))[:, None]).to(wave.dtype)[:, None, None]
        index = torch.minimum(position // self.steps, (lengths - 1).clamp_min(0)[:, None])
        condition = self.condition(hidden)
        condition = condition.gather(1, index[..., None].expand(-1, -1, condition.shape[-1])).transpose(1, 2)
        magnitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-10)
        compressed = spectrum * (magnitude + 1e-4) ** (COMPRESSION - 1)
        x = torch.stack((compressed.real, compressed.imag, torch.log(magnitude + 1e-4), condition), dim=1) * mask
        x = self.input(x) * mask
        for block in self.blocks:
            x = block(x, mask)
        correction = self.output(x) * mask
        spectrum = spectrum + torch.complex(correction[:, 0], correction[:, 1]) * (magnitude + 1e-4) ** (1 - COMPRESSION)
        return torch.istft(spectrum, self.fft, self.hop, window=self.window, center=True, length=padded.shape[-1])[:, :wave.shape[-1]]


class HeadBlock(nn.Module):
    """ConvNeXt block over [B, C, frames]; frames past an utterance's end stay zero."""

    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, 3 * channels)
        self.project = nn.Linear(3 * channels, channels)

    def forward(self, x, mask):
        y = self.project(F.gelu(self.expand(self.norm(self.depthwise(x).transpose(1, 2))))).transpose(1, 2)
        return (x + y) * mask


class SpectralHead(nn.Module):
    """Frame-rate neural refinement of the source-filter output.

    The synthesis is analysed with an STFT of ``head_fft`` points at the acoustic
    hop (one frame per acoustic frame). Per frame, a ConvNeXt stack reads every
    bin's compressed real and imaginary part together with the acoustic state and
    predicts a complex correction for every bin, scaled by the bin's own level;
    the result is inverted. Working per frame instead of per bin keeps the cost
    low (Vocos), and the pulse train as input keeps the pitch exact (Wavehax).
    The correction starts at zero: a new head is the identity."""

    def __init__(self, c):
        super().__init__()
        self.fft, self.hop, self.bins = c.head_fft, c.hop, c.head_fft // 2 + 1
        width = c.head_channels
        self.input = nn.Conv1d(2 * self.bins + c.channels, width, 1)
        self.blocks = nn.ModuleList(HeadBlock(width) for _ in range(c.head_blocks))
        self.norm = nn.LayerNorm(width)
        self.output = nn.Conv1d(width, 2 * self.bins, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("window", torch.hann_window(c.head_fft), persistent=False)

    def correct(self, real, imaginary, condition, mask):
        """real, imaginary [B, bins, frames]; condition [B, channels, frames]."""
        magnitude = torch.sqrt(real * real + imaginary * imaginary + 1e-10)
        scale = (magnitude + 1e-4) ** (COMPRESSION - 1)
        x = self.input(torch.cat((real * scale, imaginary * scale, condition), dim=1) * mask) * mask
        for block in self.blocks:
            x = block(x, mask)
        correction = self.output(self.norm(x.transpose(1, 2)).transpose(1, 2)) * mask
        level = (magnitude + 1e-4) ** (1 - COMPRESSION)
        return real + correction[:, :self.bins] * level, imaginary + correction[:, self.bins:] * level

    def forward(self, wave, hidden, lengths):
        # Half a window of trailing zeros; frames reaching into it use the last real
        # frame's state, frames further out are masked: batching never changes audio.
        padded = F.pad(wave, (0, self.fft // 2))
        spectrum = torch.stft(padded, self.fft, self.hop, window=self.window, center=True, pad_mode="constant", return_complex=True)
        position = torch.arange(spectrum.shape[-1], device=wave.device)[None]
        mask = (position <= (lengths + self.fft // (2 * self.hop))[:, None]).to(wave.dtype)[:, None]
        index = torch.minimum(position, (lengths - 1).clamp_min(0)[:, None])
        condition = hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1])).transpose(1, 2)
        real, imaginary = self.correct(spectrum.real, spectrum.imag, condition, mask)
        return torch.istft(torch.complex(real, imaginary), self.fft, self.hop, window=self.window, center=True,
                           length=padded.shape[-1])[:, :wave.shape[-1]]


class SourceFilter(nn.Module):
    """Harmonic-plus-noise synthesis controlled per frame by two linear heads."""

    def __init__(self, c):
        super().__init__()
        self.config = c
        self.harmonic_steps, self.noise_steps = c.hop // c.synth_hop, c.hop // c.noise_hop
        self.harmonic = nn.Linear(c.channels, c.cep_causal + c.cep_anti + self.harmonic_steps)
        self.noise = nn.Linear(c.channels, c.noise_cep + self.noise_steps)
        for head, level in ((self.harmonic, -1.5), (self.noise, -2.5)):   # start quiet and spectrally flat
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            head.bias.data[0] = level

        def lifter(size, first):
            scale = 1.0 / (1.0 + torch.arange(first, first + size, dtype=torch.float32) / 4)
            if first == 0:
                scale[0] = 2.0      # coefficient 0 is the log gain and needs range
            return scale
        self.register_buffer("causal_lifter", lifter(c.cep_causal, 0), persistent=False)
        self.register_buffer("anti_lifter", lifter(c.cep_anti, 1), persistent=False)
        self.register_buffer("noise_lifter", lifter(c.noise_cep, 0), persistent=False)
        self.register_buffer("synth_window", torch.hann_window(c.synth_fft), persistent=False)
        self.register_buffer("noise_window", torch.hann_window(c.noise_fft), persistent=False)
        self.postfilter = SpectralPostFilter(c) if c.postfilter_channels else None
        self.head = SpectralHead(c) if c.head_channels else None

    @staticmethod
    def _take(values, index):
        return values.gather(1, index[..., None].expand(-1, -1, values.shape[-1]))

    def _subframes(self, frames, lengths, steps):
        """Sub-frame j uses frame j // steps; indices past an utterance's end
        repeat its last sub-frame so padding never changes real audio. Two extra
        sub-frames cover the half window of zeros appended to the source."""
        sub = torch.arange(steps * frames + 3, device=lengths.device)[None].expand(len(lengths), -1)
        return torch.minimum(sub, (steps * lengths - 1).clamp_min(0)[:, None])

    @staticmethod
    def _shape(source, response, fft, hop, window):
        # Half a window of trailing zeros: the frames around an utterance's end are
        # then the same whether it stands alone or is padded inside a batch.
        return filtered(F.pad(source, (0, fft // 2)), response, fft, hop, window)[:, :source.shape[-1]]

    def forward(self, hidden, f0, lengths, noise=None, phase=None, voiced=None):
        c = self.config
        frames = hidden.shape[1]
        samples = frames * c.hop
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.float()
            audible = length_mask(lengths * c.hop, samples)
            last = (lengths - 1).clamp_min(0)[:, None]
            # Harmonic branch: envelope interpolated between frames, one gain per sub-frame.
            h = self.harmonic(hidden)
            split = c.cep_causal + c.cep_anti
            sub = self._subframes(frames, lengths, self.harmonic_steps)
            here, following = sub // self.harmonic_steps, torch.minimum((sub + 1) // self.harmonic_steps, last)
            causal = 0.5 * (self._take(h[..., :c.cep_causal], here) + self._take(h[..., :c.cep_causal], following)) * self.causal_lifter
            anti = 0.5 * (self._take(h[..., c.cep_causal:split], here) + self._take(h[..., c.cep_causal:split], following)) * self.anti_lifter
            causal = torch.cat((causal[..., :1] + h[..., split:].flatten(1).gather(1, sub)[..., None], causal[..., 1:]), dim=-1)
            harmonic_response = cepstrum_response(causal, anti, c.synth_fft)
            # Noise branch: one envelope per frame, four gains per frame for sharp consonants.
            n = self.noise(hidden)
            sub = self._subframes(frames, lengths, self.noise_steps)
            shape = self._take(n[..., :c.noise_cep], sub // self.noise_steps) * self.noise_lifter
            shape = torch.cat((shape[..., :1] + n[..., c.noise_cep:].flatten(1).gather(1, sub)[..., None], shape[..., 1:]), dim=-1)
            noise_response = cepstrum_response(shape, None, c.noise_fft)
            # Hold the last real pitch value through padding: the pitch glide inside an
            # utterance's final frame must not depend on what the batch was padded with.
            f0 = f0.float().gather(1, torch.minimum(torch.arange(frames, device=f0.device)[None], last))
            if noise is None:
                noise = torch.randn(hidden.shape[0], samples, device=hidden.device)
            offset = jitter_offset(noise * audible, c.excitation_jitter) if c.excitation_jitter > 0 else None
            pulses = pulse_train(f0, c.sample_rate, c.hop, c.harmonic_limit, phase, offset) * audible
            if c.unvoiced_pulse_gain < 1 and voiced is not None:
                # Hold the last real voicing value through padding, like the pitch above.
                voiced = voiced.float().gather(1, torch.minimum(torch.arange(frames, device=voiced.device)[None], last))
                pulses = pulses * pulse_gate(voiced, c.hop, c.unvoiced_pulse_gain)
            wave = (self._shape(pulses, harmonic_response, c.synth_fft, c.synth_hop, self.synth_window)
                    + self._shape(noise * audible, noise_response, c.noise_fft, c.noise_hop, self.noise_window))
            if self.postfilter is not None:
                wave = self.postfilter(wave * audible, hidden, lengths)
            if self.head is not None:
                wave = self.head(wave * audible, hidden, lengths)
            return wave * audible


class MelCorrection(nn.Module):
    """Residual correction of the Vocos log-mel of the source-filter synthesis.

    ``mel = dsp_mel + mask * Linear([acoustic state, dsp_mel])``, zero-initialised so it
    starts exactly at the uncorrected spectrogram. The mask tapers it in above
    ``low_hz`` (default 1 kHz): blind band-swap tests (quality-journey §6.3) put the
    metallic trace above 1 kHz, with the lower bands and their pitch detail fine.
    It sees the DSP mel as well as the acoustic state, so it can fix band texture,
    not only the smooth envelope."""

    def __init__(self, channels, bands=100, low_hz=1000.0, sample_rate=24000, fft=1024):
        super().__init__()
        from .losses import VocosMel
        self.project = nn.Linear(channels + bands, bands)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        filters = VocosMel(sample_rate, fft, 256, bands).filters
        centres = (filters * torch.linspace(0, sample_rate / 2, fft // 2 + 1)).sum(1) / filters.sum(1).clamp_min(1e-9)
        ramp = ((centres - 0.8 * low_hz) / (0.4 * low_hz)).clamp(0, 1)
        self.register_buffer("mask", ramp * ramp * (3 - 2 * ramp), persistent=False)

    def forward(self, acoustic, dsp_mel):
        """acoustic [B, T, channels], dsp_mel [B, T + 1, bands] -> corrected [B, T + 1, bands]."""
        frames = acoustic.shape[1]
        correction = self.project(torch.cat((acoustic.float(), dsp_mel[:, :frames]), dim=-1))
        correction = torch.cat((correction, correction[:, -1:]), dim=1)       # the extra centred frame
        return dsp_mel + self.mask * correction


class TinyVoca(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = c = config
        self.tokens = nn.Embedding(c.vocab_size, c.channels, padding_idx=0)
        self.stress = nn.Embedding(3, c.channels)
        self.voices = nn.Embedding(len(c.speakers), c.channels)
        self.encoder = Stack(c, c.encoder_layers)
        # Outputs per token: log(1 + frames) and, for pause-capable tokens, a
        # "has frames at all" logit. A pause is absent more often than present, so
        # one regression would average both cases into a pause that is too short.
        self.duration = Predictor(c, 2)
        self.frame_voices = nn.Embedding(len(c.speakers), c.channels)
        if c.emotions:
            # Scaled by intensity, so intensity 0 (and every unlabelled corpus) adds nothing.
            self.emotions = nn.Embedding(len(c.emotions), c.channels)
            self.frame_emotions = nn.Embedding(len(c.emotions), c.channels)
            for table in (self.emotions, self.frame_emotions):
                nn.init.normal_(table.weight, std=0.02)
        else:
            self.emotions = self.frame_emotions = None
        self.frame_position = nn.Linear(3, c.channels)
        self.prosody_stack = Stack(c, c.prosody_layers)
        # Outputs: continuous log(F0/reference), voiced logit, normalized log RMS energy.
        self.prosody_head = nn.Linear(c.channels, 3)
        self.prosody_projection = nn.Linear(3, c.channels)
        self.acoustic_stack = Stack(c, c.acoustic_layers)
        self.vocoder = SourceFilter(c)
        # Optional log-mel output for a neural vocoder (Vocos convention, see Config).
        self.mel_output = nn.Linear(c.channels, c.vocos_mels) if c.vocos_mels else None
        # Token classes come from this model's own language profile, never from
        # process-wide state: models of different languages can coexist.
        profile = c.profile
        phones = torch.zeros(c.vocab_size, dtype=torch.bool)
        phones[sorted(profile.phone_ids)] = True
        self.register_buffer("phone_tokens", phones, persistent=False)
        vowels = torch.zeros(c.vocab_size, dtype=torch.bool)
        vowels[sorted(profile.vowel_ids)] = True
        self.register_buffer("vowel_tokens", vowels, persistent=False)

    def emotion_vector(self, table, emotions, intensities, batch, like):
        """[B, channels] contribution of the emotion input (zeros without one)."""
        if table is None or emotions is None:
            return torch.zeros(batch, like.shape[-1], device=like.device, dtype=like.dtype)
        return table(emotions) * intensities.to(like.dtype)[:, None]

    def encode(self, tokens, stress, lengths, speakers, emotions=None, intensities=None):
        if tokens.ndim != 2 or stress.shape != tokens.shape or lengths.shape != (tokens.shape[0],) or speakers.shape != lengths.shape:
            raise ValueError("expected tokens[B,N], stress[B,N], lengths[B], speakers[B]")
        if tokens.shape[1] > self.config.max_tokens or bool((lengths < 1).any()) or bool((lengths > tokens.shape[1]).any()):
            raise ValueError("invalid token length")
        mask = length_mask(lengths, tokens.shape[1])
        x = self.tokens(tokens) + self.stress(stress) + self.voices(speakers)[:, None]
        x = (x + self.emotion_vector(self.emotions, emotions, intensities, len(lengths), x)[:, None]) * mask[..., None]
        return self.encoder(x, mask), mask

    def expand(self, hidden, durations, token_mask, speakers, emotions=None, intensities=None):
        """Repeat token states by integer durations (zero is allowed for pause
        tokens) and add each frame's position inside its token."""
        if durations.shape != token_mask.shape or bool((durations < 0).any()) or bool((durations[~token_mask] != 0).any()):
            raise ValueError("invalid durations")
        lengths = durations.sum(1)
        if bool((lengths < 1).any()) or bool((lengths > self.config.max_frames).any()):
            raise ValueError("utterance has no frames or exceeds max_frames; split the input")
        size = int(lengths.max())
        ends = durations.cumsum(1)
        frame = torch.arange(size, device=hidden.device)[None].expand(hidden.shape[0], size).contiguous()
        index = torch.searchsorted(ends, frame, right=True).clamp(max=durations.shape[1] - 1)
        owned = durations.gather(1, index).clamp_min(1)
        offset = frame - (ends.gather(1, index) - durations.gather(1, index))
        relative = (offset.to(hidden.dtype) + 0.5) / owned
        features = torch.stack((relative, 1 - relative, torch.log1p(owned.to(hidden.dtype)) / 4), dim=-1)
        mask = frame < lengths[:, None]
        expanded = hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))
        expanded = expanded + self.frame_position(features) + self.frame_voices(speakers)[:, None]
        expanded = expanded + self.emotion_vector(self.frame_emotions, emotions, intensities, len(lengths), expanded)[:, None]
        return expanded * mask[..., None], mask, lengths

    def predict_prosody(self, frames, frame_mask):
        hidden = self.prosody_stack(frames, frame_mask)
        return hidden, self.prosody_head(hidden) * frame_mask[..., None]

    @staticmethod
    def hard_prosody(prediction):
        """What the decoder is conditioned on when it is not teacher-forced."""
        return torch.stack((prediction[..., 0].clamp(-2.0, 1.5), (prediction[..., 1] > 0).to(prediction.dtype),
                            prediction[..., 2].clamp(-4.5, 3.0)), dim=-1)

    def decode(self, prosody_hidden, condition, frame_mask):
        return self.acoustic_stack((prosody_hidden + self.prosody_projection(condition)) * frame_mask[..., None], frame_mask)

    def frequency(self, log_f0):
        c = self.config
        return (c.pitch_reference_hz * torch.exp(log_f0.float())).clamp(c.f0_min, c.f0_max)

    def predict_durations(self, hidden, mask, tokens, speed=1.0):
        output = self.duration(hidden, mask).float()
        frames = torch.round(torch.expm1(output[..., 0].clamp(0, math.log1p(400))) / speed).long().clamp_min(1)
        # Every phoneme is audible for at least one frame. A pause-capable token
        # (edge, punctuation, word boundary) lasts its predicted length when the
        # presence logit says there is a pause, and vanishes otherwise.
        present = self.phone_tokens[tokens] | (output[..., 1] > 0)
        return torch.where(present, frames, torch.zeros_like(frames)) * mask

    @staticmethod
    def widen_pitch(prosody, frame_mask, pitch_range):
        """Scale the predicted pitch movement around each utterance's voiced mean.
        Regression predicts a flatter contour than the recordings; values above 1
        restore range (1.0 leaves the prediction untouched)."""
        if pitch_range == 1.0:
            return prosody
        voiced = ((prosody[..., 1] > 0) & frame_mask).to(prosody.dtype)
        mean = (prosody[..., 0] * voiced).sum(1, keepdim=True) / voiced.sum(1, keepdim=True).clamp_min(1)
        pitch = mean + (prosody[..., 0] - mean) * pitch_range
        return torch.cat((pitch[..., None], prosody[..., 1:]), dim=-1) * frame_mask[..., None]

    @staticmethod
    def smooth_pitch(pitch, voiced, frame_mask, frames=PITCH_SMOOTH_FRAMES):
        """Light low-pass of the predicted log-F0 contour [B, T] (Hann window of ``frames``),
        then the voiced spread is restored so the intonation range is unchanged. The
        predicted contour jitters frame to frame more than recorded pitch (heard as a
        tremble, worst on EF3: 0.38 vs 0.28 semitones per frame); 5 frames matches the
        recordings. Masked, so padding and batching do not change a row."""
        if frames <= 1:
            return pitch
        window = torch.hann_window(frames + 2, periodic=False, dtype=pitch.dtype, device=pitch.device)[1:-1]
        window = (window / window.sum()).view(1, 1, -1)
        mask = frame_mask.to(pitch.dtype)[:, None]
        smoothed = (F.conv1d(pitch[:, None] * mask, window, padding=frames // 2)
                    / F.conv1d(mask, window, padding=frames // 2).clamp_min(1e-6))[:, 0]
        weight = voiced.to(pitch.dtype) * frame_mask.to(pitch.dtype)
        count = weight.sum(1, keepdim=True).clamp_min(1)
        mean = (smoothed * weight).sum(1, keepdim=True) / count
        spread = lambda x, m: (((x - m) ** 2 * weight).sum(1, keepdim=True) / count).sqrt()
        original = (pitch * weight).sum(1, keepdim=True) / count
        scale = spread(pitch, original) / spread(smoothed, mean).clamp_min(1e-6)
        return torch.where(frame_mask, mean + (smoothed - mean) * scale, pitch)

    @staticmethod
    def pitch_wobble(shape, semitones, seed, device):
        """Small random pitch movement for the oscillator only (log-F0 units).
        Recorded pitch has cycle-to-cycle irregularity that a predicted contour
        lacks; two thirds of the wobble is slow drift, one third frame-level."""
        generator = torch.Generator().manual_seed(seed + 7919)
        fast = torch.randn(shape, generator=generator)
        slow = F.avg_pool1d(torch.randn(shape[0], 1, shape[1] + 16, generator=generator), 17, stride=1)[:, 0]
        slow = slow / slow.std().clamp_min(1e-6)
        return ((2 * slow + fast) / 5 ** 0.5 * semitones * math.log(2) / 12).to(device)

    @torch.inference_mode()
    def synthesize(self, tokens, stress, lengths, speakers, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, pitch_jitter=0.0,
                   voiced_vowels=True, emotions=None, intensities=None, pitch_smooth=PITCH_SMOOTH_FRAMES):
        """Free-running synthesis: predicted durations, pitch, voicing and energy."""
        if not math.isfinite(speed) or not 0.25 <= speed <= 4 or not math.isfinite(pitch_shift) or not 0 <= pitch_range <= 3:
            raise ValueError("speed must be between 0.25 and 4, pitch_shift finite and pitch_range between 0 and 3")
        hidden, mask = self.encode(tokens, stress, lengths, speakers, emotions, intensities)
        durations = self.predict_durations(hidden, mask, tokens, speed)
        frames, frame_mask, frame_lengths = self.expand(hidden, durations, mask, speakers, emotions, intensities)
        prosody_hidden, prosody = self.predict_prosody(frames, frame_mask)
        if voiced_vowels:
            # The pitch tracker missed creaky vowels in the recordings, and the
            # voicing predictor learned to drop whole vowels (heard as a whispered
            # word). A vowel is voiced by definition, so its frames are kept voiced.
            ends = durations.cumsum(1)
            frame = torch.arange(frames.shape[1], device=tokens.device)[None].expand(len(lengths), -1).contiguous()
            owner = torch.searchsorted(ends, frame, right=True).clamp(max=tokens.shape[1] - 1)
            vowel = self.vowel_tokens[tokens].gather(1, owner) & frame_mask
            prosody = torch.cat((prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None],
                                 prosody[..., 2:]), dim=-1)
        if pitch_smooth > 1:
            smoothed = self.smooth_pitch(prosody[..., 0], prosody[..., 1] > 0, frame_mask, pitch_smooth)
            prosody = torch.cat((smoothed[..., None], prosody[..., 1:]), dim=-1)
        prosody = self.widen_pitch(prosody, frame_mask, pitch_range)
        acoustic = self.decode(prosody_hidden, self.hard_prosody(prosody), frame_mask)
        # One noise stream per row, drawn at the row's own length, so batching or
        # padding never changes a row's audio.
        noise = nn.utils.rnn.pad_sequence([torch.randn(int(n) * self.config.hop, generator=torch.Generator().manual_seed(seed + row))
                                           for row, n in enumerate(frame_lengths)], batch_first=True)
        pitch = prosody[..., 0] + pitch_shift
        if pitch_jitter > 0:
            pitch = pitch + self.pitch_wobble(pitch.shape, pitch_jitter, seed, pitch.device)
        wave = self.vocoder(acoustic, self.frequency(pitch), frame_lengths, noise=noise.to(hidden.device),
                            voiced=(prosody[..., 1] > 0).float())
        details = {"durations": durations, "prosody": prosody, "frame_mask": frame_mask, "acoustic": acoustic}
        if self.mel_output is not None:
            details["mel"] = self.mel_output(acoustic) * frame_mask[..., None]
        return wave, frame_lengths * self.config.hop, details
