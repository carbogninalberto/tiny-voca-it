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
    {"state": "done", "title": "Public benchmark and vocoder over a range of pulse shares", "when": "19 Nov", "text": "MiniMax Italian test set (this page); vocoder fine-tuned so every per-voice setting is in distribution."},
    {"state": "done", "title": "Streaming", "when": "19 Nov", "text": "Sentence-level streaming in the browser: playback starts after the first sentence while the rest is synthesised. Next: chunked streaming inside long sentences, and a Python streaming API."},
    {"state": "done", "title": "New voices Lorenzo and Chiara, much stronger emotions", "when": "19 Nov", "text": "Two designed voices with 5 h of speech each, 150 emotion-congruent sentences per emotion (5× more), a voice-similarity filter, and a pitch-change loss against tremble. Data generation now ~2.4× faster (CUDA graphs)."},
    {"state": "done", "title": "Smaller downloads: lean and quantized ONNX", "when": "19 Nov", "text": "Signal-processing tables computed in the graph (47 → 28 MB); the page now loads an int8 version of 8.6 MB with the same intelligibility. fp16 (14.6 MB) and int4 (5.9 MB) variants are also built."},
    {"state": "done", "title": "Numbers read the Italian way", "when": "19 Nov", "text": "Cardinals, decimals, percentages, euro amounts, times, dates and ordinals become words before phonemisation, identically in Python and in the browser."},
    {"state": "done", "title": "Releases 0.2.0 and 0.3.0", "when": "19 Nov", "text": "Checkpoints, ONNX in four precisions (fp32 28 MB, fp16 15 MB, int8 9 MB, int4 6 MB), model card, samples and checksums on GitHub Releases."},
    {"state": "done", "title": "Clearer highs: texture loss above 2 kHz", "when": "19 Nov", "text": "A loss that asks for the recorded amount of high-band detail instead of its average: detail 0.88 → 0.93 (Lorenzo), 0.91 → 0.95 (Chiara); preferred blind on both voices. A 26% larger model gained only ~1% and was not adopted."},
    {"state": "now", "title": "tiny-voca-it-zs: zero-shot voice cloning", "when": "pilot running", "text": "A separate, bigger model (about 8–10M parameters) with its own neural voice-fingerprint network: "
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
    ours = {"system": "tiny-voca-it (Lorenzo + Chiara)", "short": "tiny-voca-it", "params": "6.8M", "wer": r["wer"], "sim": None, "ours": True,
            "wer_ci": r.get("wer_ci95"), "source": "measured, this repository (fixed voices, not cloning)"}
    rows = sorted(PUBLISHED + [ours], key=lambda x: x["wer"])
    site = {
        "updated": datetime.now(ZoneInfo("Europe/Rome")).strftime("%Y-%m-%d %H:%M") + " CET",
        "pill": "Large-model intelligibility at 6.8M parameters",
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
                f"WER by voice: Lorenzo {r['wer_by_voice']['EM4']:.2f}%, Chiara {r['wer_by_voice']['EF4']:.2f}%. Without the {r['lines_with_digits']} lines that contain digits: {r['wer_without_digits']:.2f}% "
                f"(lines with digits: {r['wer_lines_with_digits']:.2f}%; numbers are read as words since 19 November, before: {r['previous']['wer_lines_with_digits']:.2f}%). Previous version ({r['previous']['model']}): {r['previous']['wer']:.2f}%.",
                "WER measures intelligibility only: every word understood. It says nothing about naturalness, intonation or the slight metallic "
                "trace, where the 0.6–2B models are clearly ahead; no paper reports an Italian listening (MOS) score on this set.",
                f"Our 95% interval ({r['wer_ci95'][0]:.2f}–{r['wer_ci95'][1]:.2f}%, bootstrap over the 100 lines, {r['words']} words) overlaps several systems: "
                "statistically the same band, not a clear first place. None of the 164 benchmark sentences is in our training texts.",
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
