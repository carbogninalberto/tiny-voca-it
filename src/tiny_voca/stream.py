"""Streaming synthesis: yield audio sentence by sentence, so playback can start after the first one.

    from tiny_voca.stream import Streamer
    streamer = Streamer("runs/voice-two/inference-115000.pt", "runs/vocoder-three/vocoder-voicegain.pt")
    for chunk in streamer.stream("Prima frase. Seconda frase.", voice="EF3"):
        play(chunk)            # float32 numpy array at streamer.sample_rate

From the shell, raw 16-bit PCM to a player:

    python -m tiny_voca.stream --checkpoint ... --vocoder ... --voice IT1 --text "..." | aplay -r 24000 -f S16_LE -c 1

The model is non-autoregressive, so the unit of streaming is an utterance: text is split at sentence
punctuation (and, for very long sentences, at commas), as the browser playground does. Every chunk
after the first starts with 0.25 s of silence, the same pause the playground inserts.
"""

import argparse
import re
import sys
import time

import numpy as np
import torch

from .infer import load_model, match_frontend, speak, vocode

PAUSE_SECONDS = 0.25
MAX_CHARACTERS = 220        # longer sentences are split at commas, semicolons or colons


def utterances(text):
    """Sentences, and long sentences split at clause punctuation."""
    parts = []
    for sentence in re.split(r"(?<=[.!?…;])\s+", re.sub(r"\s+", " ", text).strip()):
        if not re.search(r"\w", sentence):
            continue
        if len(sentence) <= MAX_CHARACTERS:
            parts.append(sentence)
            continue
        clause = ""
        for piece in re.split(r"(?<=[,:—])\s+", sentence):
            if clause and len(clause) + len(piece) + 1 > MAX_CHARACTERS:
                parts.append(clause); clause = piece
            else:
                clause = f"{clause} {piece}".strip()
        if clause:
            parts.append(clause)
    return parts


class Streamer:
    def __init__(self, checkpoint, vocoder=None, device="cpu", threads=None):
        from .vocoder import load_vocoder
        if threads:
            torch.set_num_threads(threads)
        match_frontend(checkpoint)
        self.model = load_model(checkpoint, device).eval()
        self.vocoder = load_vocoder(vocoder, device) if vocoder else None
        self.sample_rate = self.model.config.sample_rate

    def stream(self, text, voice=None, speed=1.0, pitch_shift=0.0, pitch_range=1.0, emotion=None, intensity=0.0, seed=0):
        """Yield one float32 array per utterance (silence-led after the first)."""
        pause = np.zeros(int(PAUSE_SECONDS * self.sample_rate), dtype=np.float32)
        for index, part in enumerate(utterances(text)):
            torch.manual_seed(seed + index)
            wave, _, details = speak(self.model, part, voice, speed, pitch_shift, seed=seed + index, pitch_range=pitch_range,
                                     emotion=emotion, intensity=intensity if emotion else 0.0)
            audio = vocode(self.model, wave, details, self.vocoder, voice or self.model.config.speakers[0]) if self.vocoder else wave
            audio = audio.float().numpy()
            yield audio if index == 0 else np.concatenate((pause, audio))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocoder")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--gain", type=float, default=3.0, help="Linear gain before 16-bit conversion (peaks are clipped at full scale)")
    args = parser.parse_args()
    streamer = Streamer(args.checkpoint, args.vocoder, args.device, args.threads)
    start = time.perf_counter()
    for index, chunk in enumerate(streamer.stream(args.text, args.voice, args.speed)):
        if index == 0:
            print(f"first audio after {time.perf_counter() - start:.3f} s", file=sys.stderr)
        sys.stdout.buffer.write((np.clip(chunk * args.gain, -1, 1) * 32767).astype("<i2").tobytes())
        sys.stdout.buffer.flush()
    print(f"done after {time.perf_counter() - start:.3f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
