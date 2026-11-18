from dataclasses import asdict, dataclass
import hashlib
import json

CHECKPOINT_FORMAT = 4  # 2-3: byte tokens + learned aligner (rejected explicitly).
ANNOTATION_VERSION = 3


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# Architecture-only fields that may change when a run is branched (--init-from):
# tensors that grow keep their old values in the leading block (load_compatible).
BRANCHABLE_FIELDS = frozenset({"channels", "heads", "feedforward", "kernel", "encoder_layers", "prosody_layers", "acoustic_layers",
                               "dropout", "predictor_channels", "speakers", "max_tokens", "max_frames", "max_parameters",
                               "excitation_jitter", "postfilter_channels", "postfilter_blocks",
                               "head_channels", "head_blocks", "vocos_mels",
                               "vocos_mel_loss", "emotions"})


@dataclass(frozen=True)
class Config:
    # Acoustic features (unchanged since feature schema 2; existing files are reused).
    sample_rate: int = 24000
    hop: int = 256
    n_fft: int = 1024
    n_mels: int = 80
    f0_min: float = 50.0
    f0_max: float = 600.0
    speakers: tuple[str, ...] = ("F2", "M2")
    # Emotion labels (empty = no emotion input). Each utterance adds intensity x a learned
    # vector for its label; recorded corpora are the first label (neutral) at intensity 0.
    emotions: tuple[str, ...] = ()
    # Phoneme tokens from tiny_voca.text.SYMBOLS (front-end version 3).
    vocab_size: int = 76
    # Text encoder and frame-level transformer.
    channels: int = 192
    heads: int = 4
    feedforward: int = 640
    kernel: int = 7
    encoder_layers: int = 4
    prosody_layers: int = 2
    acoustic_layers: int = 5
    dropout: float = 0.1
    predictor_channels: int = 128
    # Source-filter waveform head: harmonic branch and noise branch STFT sizes,
    # causal/anticausal complex-cepstrum orders.
    synth_fft: int = 512
    synth_hop: int = 128
    noise_fft: int = 256
    noise_hop: int = 64
    cep_causal: int = 80
    cep_anti: int = 16
    noise_cep: int = 32
    harmonic_limit: float = 0.47
    # Standard deviation of a fast random log-F0 perturbation of the pulse train
    # (smoothed over about 1 ms). A perfectly periodic source is more harmonic than
    # a real voice between 1 and 5 kHz, which is heard as buzz; this jitter
    # spreads the upper harmonics the way cycle-to-cycle irregularity does.
    excitation_jitter: float = 0.0
    # Learned spectral post-filter on the source-filter output (0 = none): 2D
    # ConvNeXt blocks over the (frequency, time) STFT of the synthesis predict a
    # complex correction per bin, for the phase spread and harmonic fine structure
    # a smooth envelope over a pulse train cannot produce (after Wavehax, 2024).
    postfilter_channels: int = 0
    postfilter_blocks: int = 2
    # Neural spectral head (0 = none): ConvNeXt blocks over acoustic frames read
    # the STFT (head_fft, hop) of the source-filter output and the acoustic state,
    # and predict a complex correction for every bin of that STFT, which is then
    # inverted (Vocos-style iSTFT head with the synthesis as harmonic prior, as in
    # Wavehax). It can give each pitch cycle the variation a pulse train lacks.
    head_channels: int = 0
    head_blocks: int = 3
    head_fft: int = 1024
    # Bands of a log-mel output in the Vocos 24 kHz convention (0 = none; 100 for
    # charactr/vocos-mel-24khz), predicted from the acoustic state so a neural
    # vocoder can render the speech instead of the source-filter head.
    vocos_mels: int = 0
    # Weight of an L1 loss between the Vocos log-mel of the synthesized waveform and
    # of the recording (0 = off): the spectrogram a neural vocoder reads from this
    # model's output, so training optimizes the vocoder's actual input.
    vocos_mel_loss: float = 0.0
    max_tokens: int = 512
    max_frames: int = 3000
    mel_mean: float = -5.0
    mel_scale: float = 5.0
    energy_mean: float = -4.0
    energy_scale: float = 2.0
    pitch_reference_hz: float = 200.0
    # Language profile (tiny_voca.languages). "en" is the legacy contract and is
    # left out of to_dict(), so English configs and fingerprints are unchanged.
    language: str = "en"
    # Inference-parameter budget enforced by training. Raising it is a recorded
    # decision in the configuration; at the default it is left out of to_dict().
    max_parameters: int = 5_000_000

    def __post_init__(self):
        symbols = self.profile.symbols
        if self.vocab_size != len(symbols):
            raise ValueError(f"vocab_size must equal the {self.language!r} phoneme inventory size {len(symbols)}")
        if self.channels % self.heads or (self.channels // self.heads) % 2:
            raise ValueError("channels must be divisible by heads with an even head width")
        if len(set(self.emotions)) != len(self.emotions):
            raise ValueError("emotion labels must be unique")
        if not self.speakers or len(set(self.speakers)) != len(self.speakers):
            raise ValueError("speaker names must be unique and nonempty")
        for fft, hop in ((self.synth_fft, self.synth_hop), (self.noise_fft, self.noise_hop)):
            if fft != 4 * hop or self.hop % hop:
                raise ValueError("synthesis windows need 4x overlap and a hop dividing the acoustic hop")
        if self.cep_causal + self.cep_anti >= self.synth_fft // 2 or self.noise_cep >= self.noise_fft // 2:
            raise ValueError("cepstrum orders must be well below half the synthesis FFT size")
        if min(self.encoder_layers, self.prosody_layers, self.acoustic_layers, self.kernel) < 1 or not self.kernel % 2:
            raise ValueError("layer counts must be positive and the kernel odd")
        if self.vocos_mel_loss < 0 or (self.vocos_mel_loss and (self.sample_rate, self.hop) != (24000, 256)):
            raise ValueError("vocos_mel_loss must be >= 0 and needs 24 kHz audio with hop 256")
        if self.vocos_mels < 0 or (self.vocos_mels and (self.sample_rate, self.hop, self.n_fft) != (24000, 256, 1024)):
            raise ValueError("vocos_mels needs 24 kHz audio with hop 256 and n_fft 1024")
        if self.head_channels < 0 or self.head_blocks < 1 or self.head_fft % self.hop or self.head_fft < 2 * self.hop:
            raise ValueError("head_channels must be >= 0, head_blocks >= 1 and head_fft a multiple (>= 2) of hop")
        if self.postfilter_channels < 0 or self.postfilter_blocks < 1:
            raise ValueError("postfilter_channels must be >= 0 and postfilter_blocks >= 1")
        if not 0 <= self.excitation_jitter <= 0.2:
            raise ValueError("excitation_jitter must be between 0 and 0.2")
        if min(self.mel_scale, self.energy_scale) <= 0 or not 0 < self.harmonic_limit < 0.5:
            raise ValueError("invalid normalization scale or harmonic limit")
        if self.max_parameters < 1:
            raise ValueError("max_parameters must be positive")

    @property
    def profile(self):
        from .languages import profile_for
        return profile_for(self.language)

    def to_dict(self):
        """Fields added after the format-4 release are written only when they
        differ from their default, so older configs, fingerprints and
        checkpoints stay byte-identical."""
        value = asdict(self)
        for key, default in OMITTED_AT_DEFAULT.items():
            if value[key] == default:
                del value[key]
        return value

    def branch_differences(self, source):
        """Fields that differ from ``source`` (another Config). Raises if a
        difference is not allowed when branching with --init-from: signal
        processing, vocabulary and language must match, and existing voices
        must keep their positions (new voices may only be appended)."""
        differences = {key: (getattr(source, key), getattr(self, key)) for key in asdict(self) if getattr(source, key) != getattr(self, key)}
        fixed = sorted(set(differences) - BRANCHABLE_FIELDS)
        if fixed:
            raise ValueError(f"cannot branch across a change of {fixed}: only {sorted(BRANCHABLE_FIELDS)} may differ")
        if self.emotions[:len(source.emotions)] != source.emotions:
            raise ValueError(f"existing emotions {list(source.emotions)} must stay first and in order; new ones are appended")
        if self.speakers[:len(source.speakers)] != source.speakers:
            raise ValueError(f"existing voices {list(source.speakers)} must stay first and in order; new voices are appended")
        return differences

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "speakers": tuple(value["speakers"]), "emotions": tuple(value.get("emotions", ()))})

    @classmethod
    def for_language(cls, language, speakers, **overrides):
        """Default architecture with a language's vocabulary and the given voices."""
        from .languages import profile_for
        return cls(language=language, speakers=tuple(speakers), vocab_size=len(profile_for(language).symbols), **overrides)

    @property
    def fingerprint(self):
        return _digest(self.to_dict())

    @property
    def acoustic_config(self):
        """Settings that determine stored wave/mel/prosody tensors. Tokens and
        durations are versioned separately by the annotation layer."""
        fields = ("sample_rate", "hop", "n_fft", "n_mels", "f0_min", "f0_max")
        return {**{key: getattr(self, key) for key in fields}, "feature_version": 2,
                "pitch_reference_hz": 200, "log_floor": 1e-5}

    @property
    def acoustic_fingerprint(self):
        return _digest(self.acoustic_config)

    @property
    def legacy_feature_fingerprint(self):
        """Fingerprint written by the byte-token preparation (front end 2).

        Feature files carrying it hold compatible acoustics and STALE tokens;
        only the annotation step may read them, and it ignores their tokens.
        """
        fields = ("sample_rate", "hop", "n_fft", "n_mels", "f0_min", "f0_max", "speakers")
        return _digest({**{key: getattr(self, key) for key in fields}, "vocab_size": 259, "feature_version": 2,
                        "frontend_version": 2, "pitch_reference_hz": 200, "log_floor": 1e-5})


OMITTED_AT_DEFAULT = {"language": "en", "max_parameters": 5_000_000, "excitation_jitter": 0.0,
                      "postfilter_channels": 0, "postfilter_blocks": 2,
                      "head_channels": 0, "head_blocks": 3, "head_fft": 1024,
                      "vocos_mels": 0, "vocos_mel_loss": 0.0, "emotions": ()}
