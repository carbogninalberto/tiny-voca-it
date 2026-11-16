"""Local playground: type text, hear a tiny-voca checkpoint speak it.

    python scripts/playground.py --checkpoint runs/it-base/inference.pt
    python scripts/playground.py --checkpoint base=runs/it-base/inference.pt --checkpoint mine=runs/it-adapt/inference.pt
    # then open http://127.0.0.1:8765

Synthesis runs in Python (the Italian front end needs eSpeak NG, which the
browser cannot run). Only the standard library serves HTTP; nothing leaves the
machine. Long text is split into sentences and joined with short pauses.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import sys
import threading
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_voca.infer import load_model, match_frontend, speak  # noqa: E402

PAGE = Path(__file__).resolve().parent.parent / "playground" / "index.html"
SENTENCE = re.compile(r"(?<=[.!?…;])\s+")


def sentences(text, limit=400):
    parts, current = [], ""
    for piece in SENTENCE.split(text.strip()):
        if not piece:
            continue
        if len(piece) > limit:                       # split very long sentences at commas
            for sub in re.split(r"(?<=,)\s+", piece):
                parts.append(sub)
        else:
            parts.append(piece)
    return [p for p in parts if p.strip()]


class Playground:
    """Checkpoints are reloaded when their file changes, so a playground pointed
    at a running training's inference.pt always speaks with the newest weights."""

    def __init__(self, checkpoints, device):
        self.models, self.lock, self.device = {}, threading.Lock(), device
        for name, path in checkpoints.items():
            self.models[name] = self._load(path)

    def _load(self, path):
        path = Path(path)
        for attempt in range(3):          # the trainer replaces the file atomically; retry a torn read anyway
            try:
                stamp = path.stat().st_mtime
                match_frontend(path)
                model = load_model(path, self.device)
                state = torch.load(path, map_location="cpu", weights_only=True)
                break
            except (OSError, RuntimeError, EOFError):
                if attempt == 2:
                    raise
                time.sleep(1)
        return {"model": model, "path": str(path), "step": state.get("step"), "mtime": stamp,
                "language": model.config.language, "voices": list(model.config.speakers),
                "adapted": bool(state.get("provenance", {}).get("adapted_from"))}

    def refresh(self):
        for name, entry in list(self.models.items()):
            try:
                if Path(entry["path"]).stat().st_mtime != entry["mtime"]:
                    self.models[name] = self._load(entry["path"])
            except (OSError, RuntimeError, EOFError, ValueError):
                pass                              # keep serving the previous weights

    def info(self):
        with self.lock:
            self.refresh()
            return {"checkpoints": [{k: v for k, v in {"name": name, **entry}.items() if k not in ("model", "mtime")}
                                    for name, entry in self.models.items()]}

    def speak(self, request):
        entry = self.models.get(request.get("checkpoint")) or next(iter(self.models.values()))
        model = entry["model"]
        text = str(request.get("text", "")).strip()
        if not text:
            raise ValueError("Scrivi un testo / type some text")
        if len(text) > 3000:
            raise ValueError("text too long (3000 characters max)")
        options = dict(voice=request.get("voice") or None, speed=float(request.get("speed", 1.0)),
                       pitch_shift=float(request.get("pitch_shift", 0.0)) * np.log(2) / 12,     # semitones -> log F0
                       pitch_range=float(request.get("pitch_range", 1.0)), seed=int(request.get("seed", 0)))
        pause = np.zeros(int(0.25 * model.config.sample_rate), dtype=np.float32)
        pieces, phonemes = [], []
        started = time.perf_counter()
        with self.lock, torch.inference_mode():
            self.refresh()
            entry = self.models.get(request.get("checkpoint")) or next(iter(self.models.values()))
            model = entry["model"]
            for sentence in sentences(text):
                wave, phones, _ = speak(model, sentence, **options)
                pieces += [wave.numpy(), pause]
                phonemes.append(" ".join(s for s in phones.symbols if not s.startswith("<") or s == "<sp>").replace("<sp>", "·"))
        audio = np.concatenate(pieces[:-1]) if pieces else pause
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 0.99:                              # never hard-clip; scale the whole utterance
            audio = audio * (0.99 / peak)
        buffer = io.BytesIO()
        sf.write(buffer, audio, model.config.sample_rate, format="WAV", subtype="PCM_16")
        seconds = len(audio) / model.config.sample_rate
        return buffer.getvalue(), {"seconds": round(seconds, 2), "compute_seconds": round(time.perf_counter() - started, 3),
                                   "phonemes": " | ".join(phonemes), "voice": options["voice"] or model.config.speakers[0]}


def handler(playground):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body, content_type, headers=()):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._send(200, json.dumps(playground.info()).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/speak":
                return self._send(404, b"not found", "text/plain")
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 100_000)
                audio, details = playground.speak(json.loads(self.rfile.read(length) or b"{}"))
            except (ValueError, KeyError, RuntimeError) as error:
                return self._send(400, json.dumps({"error": str(error)}).encode(), "application/json")
            self._send(200, audio, "audio/wav", [("X-Details", json.dumps(details, ensure_ascii=True))])

        def log_message(self, format, *args):
            sys.stderr.write("%s %s\n" % (self.command, self.path))

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", action="append", required=True, help="inference.pt, or name=path (repeatable)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    checkpoints = {}
    for value in args.checkpoint:
        name, _, path = value.rpartition("=")
        path = Path(path)
        checkpoints[name or path.parent.name or path.stem] = path
    playground = Playground(checkpoints, args.device)
    server = ThreadingHTTPServer((args.host, args.port), handler(playground))
    print(f"tiny-voca playground: http://{args.host}:{args.port}  ({', '.join(checkpoints)})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
