"""Write build/pages/site.json for the landing page: benchmark table (ours measured, others as published),
key numbers, and the current state / roadmap with the build timestamp.

    python scripts/site_data.py --result bench/out/result.json --output build/pages/site.json
"""
import argparse, json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Italian split of the MiniMax TTS Multilingual Test Set, as printed by each paper (WER in %, SIM cosine).
PUBLISHED = [
    {"system": "Qwen3-TTS 12Hz 1.7B Base", "short": "Qwen3-TTS 1.7B", "params": "1.7B", "wer": 0.948, "sim": 0.817, "source": "Qwen3-TTS report, Table 6"},
    {"system": "Qwen3-TTS 25Hz 1.7B Base", "short": "Qwen3-TTS 1.7B (25Hz)", "params": "1.7B", "wer": 1.105, "sim": 0.718, "source": "Qwen3-TTS report, Table 6"},
    {"system": "Qwen3-TTS 25Hz 0.6B Base", "short": "Qwen3-TTS 0.6B", "params": "0.6B", "wer": 1.169, "sim": 0.722, "source": "Qwen3-TTS report, Table 6"},
    {"system": "Fish-Audio S2", "short": "Fish-Audio S2", "params": "–", "wer": 1.27, "sim": 0.747, "source": "dots.tts report, Table 3"},
    {"system": "MiniMax-Speech", "short": "MiniMax-Speech", "params": "–", "wer": 1.543, "sim": 0.699, "source": "MiniMax-Speech paper, Table 2"},
    {"system": "VoxCPM 2", "short": "VoxCPM 2", "params": "–", "wer": 1.56, "sim": 0.780, "source": "dots.tts report, Table 3"},
    {"system": "ElevenLabs Multilingual v2", "short": "ElevenLabs v2", "params": "–", "wer": 1.743, "sim": 0.579, "source": "MiniMax-Speech paper, Table 2"},
    {"system": "dots.tts (MF4)", "short": "dots.tts", "params": "2B", "wer": 2.16, "sim": 0.843, "source": "dots.tts report, Table 3"},
]

ROADMAP = [
    {"state": "done", "title": "Neural vocoder with a harmonic prior", "when": "Nov 2026", "text": "Replaced the buzzy DSP output; phase from a pitch-driven pulse train, distilled from Vocos."},
    {"state": "done", "title": "Two product voices: IT1 and EF3", "when": "18 Nov", "text": "6 h of EF3 data generated and screened; voice model fine-tuned on both voices (EF3 mel error -33%)."},
    {"state": "done", "title": "Less metallic, no tremble", "when": "18 Nov", "text": "Retrained vocoder with a per-voice high-band pulse share; 5-frame pitch smoothing. Both preferred in blind tests."},
    {"state": "now", "title": "Benchmarks and vocoder over a range of pulse shares", "when": "now", "text": "Public Italian benchmark (this page); vocoder fine-tuned so every per-voice setting is in distribution."},
    {"state": "next", "title": "Voice model: intonation and texture", "when": "next", "text": "A pitch-delta loss (intonation instead of jitter), wider pitch range (now ~70% of real speech), a noise-tolerant loss above 2 kHz."},
    {"state": "next", "title": "Better text normalisation", "when": "planned", "text": "Numbers, dates, prices and abbreviations read the Italian way."},
    {"state": "next", "title": "Release", "when": "planned", "text": "Checkpoints, ONNX model and a model card; optional fine-tuning to a new voice from about 30 minutes of audio."},
    {"state": "next", "title": "tiny-voca-it-zs: zero-shot voice cloning", "when": "final", "text": "A separate, bigger model (about 8–10M parameters) with its own neural voice-fingerprint network: "
     "record 15 seconds and speak in that voice instantly, in the browser, with no training. Trained on hundreds of Italian voices; "
     "tiny-voca-it stays the small two-voice model."},
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", type=Path, required=True); p.add_argument("--synth", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = json.loads(a.result.read_text())
    synth = [json.loads(l) for l in (a.synth or a.result.parent / "synth.jsonl").read_text().splitlines()]
    rtf = sum(x["compute_seconds"] for x in synth) / sum(x["audio_seconds"] for x in synth)
    ours = {"system": "tiny-voca-it (IT1 + EF3)", "short": "tiny-voca-it", "params": "6.8M", "wer": r["wer"], "sim": None, "ours": True,
            "source": "measured, this repository"}
    rows = sorted(PUBLISHED + [ours], key=lambda x: x["wer"])
    site = {
        "updated": datetime.now(ZoneInfo("Europe/Rome")).strftime("%Y-%m-%d %H:%M") + " CET",
        "pill": f"{r['wer']:.2f}% WER on the public MiniMax Italian test set",
        "benchmark": {
            "ours": {"wer": r["wer"], "rtf": 1 / (r.get("cpu", {}).get("x_real_time") or 1 / rtf)},
            "kpis": [
                {"value": f"{r['wer']:.2f}%", "label": "WER, Whisper-large-v3"},
                {"value": f"{r['cer']:.2f}%", "label": "CER, Whisper-large-v3"},
                {"value": f"{r.get('cpu', {}).get('x_real_time') or round(1 / rtf)}×", "label": "real time, 1 CPU thread"},
                {"value": f"{r['gpu']['x_real_time']}×", "label": f"real time, GPU ({r['gpu']['device'].replace('NVIDIA GeForce ', '')}, batch 1)"},
            ],
            "rows": rows,
            "notes": [
                f"WER by voice: IT1 {r['wer_by_voice']['IT1']:.2f}%, EF3 {r['wer_by_voice']['EF3']:.2f}%. Without the {r['lines_with_digits']} lines that contain digits: {r['wer_without_digits']:.2f}% "
                f"(lines with digits: {r['wer_lines_with_digits']:.1f}%; our text normalisation does not expand numbers yet).",
                "Not an identical task: the other systems clone two Common Voice reference speakers zero-shot, which is harder than speaking "
                "in voices the model was trained on. Read the table as intelligibility at a given size, not as a ranking of voice cloning. "
                "SIM (similarity to the reference speaker) therefore does not apply to tiny-voca-it.",
                "Other systems are zero-shot cloning models of 0.6–2B+ parameters; their numbers are copied from the papers named in the table. "
                "Our normalisation: lower case, punctuation removed (seed-tts-eval style); papers may normalise slightly differently.",
                "Long lines are read sentence by sentence, as in the playground. Speed measured on one CPU thread, including the vocoder.",
            ],
        },
        "roadmap": ROADMAP,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(site, ensure_ascii=False, indent=1))
    print(a.output)


if __name__ == "__main__":
    main()
