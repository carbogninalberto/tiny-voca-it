import argparse
import os
from pathlib import Path
import torch
from .config import CHECKPOINT_FORMAT, Config
from .model import TinyVoca
from .text import FRONTEND_VARIABLE, FRONTENDS, frontend_fingerprint, phonemize, set_frontend


def load_model(path, device="cpu", check_frontend=True):
    """``check_frontend=False`` lets an ENGLISH checkpoint speak tokens of the
    other English front end (same inventory, slightly different conventions):
    for listening tests of the new phonemizer before the model is retrained on
    it. A checkpoint without language metadata is legacy English; a language
    profile mismatch is always an error."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format: byte-token checkpoints (format 2-3) cannot be used with this code")
    config = Config.from_dict(state["config"])
    profile = config.profile
    recorded = state.get("language")
    if profile.language != "en":
        if not recorded or recorded.get("language") != profile.language or recorded.get("profile_digest") != profile.digest:
            raise ValueError(f"checkpoint's {profile.language!r} language profile differs from this code's (digest mismatch)")
        if state["frontend"]["fingerprint"] != frontend_fingerprint(profile):
            raise ValueError("checkpoint was trained with a different phoneme inventory/front end")
    elif recorded is not None and recorded.get("language", "en") != "en":
        raise ValueError("checkpoint language metadata contradicts its configuration")
    elif check_frontend and state["frontend"]["fingerprint"] != frontend_fingerprint():
        raise ValueError("checkpoint was trained with a different phoneme inventory/front end")
    model = TinyVoca(config)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def match_frontend(path):
    """Select the English front end the checkpoint was trained with (other
    languages have exactly one front end, chosen by their profile)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["config"].get("language", "en") != "en":
        return "espeak"
    wanted = state["frontend"]["fingerprint"]
    for name in FRONTENDS:
        set_frontend(name)
        try:
            if frontend_fingerprint() == wanted:
                return name
        except ImportError:
            continue
    set_frontend(FRONTENDS[0])
    return None


def speak(model, text, voice, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, emotion=None, intensity=0.0):
    """Text -> (float waveform tensor, phonemized text). Same front end as training."""
    device = next(model.parameters()).device
    if getattr(model.config, "language", "en") == "it":
        # Numbers, times, dates, amounts to words (outside the front-end identity; see italian_numbers).
        from .italian_numbers import expand_numbers
        text = expand_numbers(text)
    phones = phonemize(text, model.config.profile)
    if len(phones) > model.config.max_tokens:
        raise ValueError("text is too long for one utterance; split it into sentences")
    tokens = torch.tensor([phones.ids], device=device)
    stress = torch.tensor([phones.stress], device=device)
    if voice is None:
        voice = model.config.speakers[0]
    if voice not in model.config.speakers:
        raise ValueError(f"unknown voice {voice!r}; this checkpoint has {list(model.config.speakers)}")
    speaker = torch.tensor([model.config.speakers.index(voice)], device=device)
    emotions = intensities = None
    if emotion is not None and model.config.emotions:
        if emotion not in model.config.emotions:
            raise ValueError(f"unknown emotion {emotion!r}; this checkpoint has {list(model.config.emotions)}")
        if not 0.0 <= intensity <= 1.5:
            raise ValueError("intensity must be between 0 and 1.5")
        emotions = torch.tensor([model.config.emotions.index(emotion)], device=device)
        intensities = torch.tensor([float(intensity)], device=device)
    audio, lengths, details = model.synthesize(tokens, stress, torch.tensor([len(phones)], device=device), speaker, speed, pitch_shift, seed,
                                               pitch_range, emotions=emotions, intensities=intensities)
    if not torch.isfinite(audio).all():
        raise RuntimeError("model produced non-finite audio")
    return audio[0, :int(lengths[0])].cpu(), phones, details


def vocode(model, wave, details, checkpoint, voice=None):
    """Render the final audio with the neural vocoder: the Vocos log-mel of the
    source-filter waveform, plus (for a prior vocoder) the model's pitch and voicing."""
    from .losses import VocosMel
    from .vocoder import Vocoder, load_vocoder
    # ``checkpoint`` may be a path or an already loaded Vocoder (streaming reuses one).
    vocoder = checkpoint if isinstance(checkpoint, Vocoder) else load_vocoder(checkpoint)
    device = next(vocoder.parameters()).device
    frames = wave.shape[-1] // model.config.hop
    with torch.no_grad():
        mel = VocosMel().to(device)(wave[None, :frames * model.config.hop].float().to(device)).transpose(1, 2)
        if not vocoder.config.prior:
            return vocoder(mel)[0]
        prosody = details["prosody"][0, :frames].to(device)
        return vocoder(mel, model.frequency(prosody[:, 0])[None].float(), (prosody[:, 1] > 0).float()[None], voice=voice)[0].cpu()


def main():
    parser = argparse.ArgumentParser(description="Generate speech from a trained tiny-voca inference checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", help="Voice name stored in the checkpoint (default: its first voice)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--pitch-shift", type=float, default=0, help="Added to log F0 (0.058 is about one semitone)")
    parser.add_argument("--pitch-range", type=float, default=1, help="Scale pitch movement around the utterance mean (1 = as predicted)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocoder", type=Path, help="Neural vocoder checkpoint (tiny_voca.vocoder) for the final audio")
    parser.add_argument("--emotion", help="Emotion label stored in the checkpoint (e.g. joy)")
    parser.add_argument("--intensity", type=float, default=1.0, help="Emotion intensity, 0 to 1.5 (with --emotion)")
    parser.add_argument("--frontend", choices=FRONTENDS, help="Phonemizer (default: TINY_VOCA_FRONTEND, else the one the checkpoint was trained with)")
    parser.add_argument("--allow-frontend-mismatch", action="store_true",
                        help="Speak although the checkpoint was trained on the other front end's tokens")
    args = parser.parse_args()
    import soundfile as sf
    if args.frontend:
        set_frontend(args.frontend)
    elif FRONTEND_VARIABLE not in os.environ:
        match_frontend(args.checkpoint)
    model = load_model(args.checkpoint, args.device, check_frontend=not args.allow_frontend_mismatch)
    if args.voice is not None and args.voice not in model.config.speakers:
        parser.error(f"unknown voice {args.voice!r}; available: {', '.join(model.config.speakers)}")
    wave, phones, details = speak(model, args.text, args.voice, args.speed, args.pitch_shift, pitch_range=args.pitch_range,
                                  emotion=args.emotion, intensity=args.intensity if args.emotion else 0.0)
    if args.vocoder:
        wave = vocode(model, wave, details, args.vocoder, args.voice or model.config.speakers[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Floating WAV preserves output for inspection; no silent clipping or normalization.
    sf.write(args.output, wave.numpy(), model.config.sample_rate, subtype="FLOAT")
    print(" ".join(phones.symbols))
    print(f"Saved {len(wave)/model.config.sample_rate:.2f}s; peak={float(wave.abs().max()):.4f}")


if __name__ == "__main__":
    main()
