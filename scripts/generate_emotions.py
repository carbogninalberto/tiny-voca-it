"""Synthetic Italian emotional speech with Qwen3-TTS (docs/analysis/emotion-cloning.md §6.2). Local GPU.

    PYTHONPATH=<dir with qwen_tts> python scripts/generate_emotions.py --root data/emotion-pilot --pilot

1. VoiceDesign 1.7B speaks one emotion-neutral reference sentence for every
   designed voice x emotion x intensity, from the voice description plus a style
   clause ("speaking very angrily, ...").
2. Base 1.7B clones each emotional reference (ICL mode, so the delivery carries
   over) on ordinary Italian sentences.
3. faster-whisper small (Italian) screens every clip; above --max-wer it is
   rejected (one retry with another seed).

No peak normalisation (loudness is part of the emotion). Resumable: finished
clips are skipped. Output: <root>/audio/*.wav, <root>/references/*.wav,
<root>/manifest.jsonl (accepted, with voice_id, emotion, intensity, wer) and
<root>/rejected.jsonl. Weights: Apache-2.0; the audio is synthetic and must be
labelled as such.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REFERENCE_TEXT = ("Oggi ho ricevuto una lettera che non mi aspettavo, e l'ho letta due volte "
                  "prima di capire che cosa volesse dire davvero.")
VOICES = {   # id: description (Italian native speakers, clean close-microphone recording)
    "EF1": "A woman in her thirties with a warm, medium-pitched voice, native Italian speaker from central Italy, clear diction, close microphone, quiet room.",
    "EM1": "A man in his forties with a deep, calm voice, native Italian speaker from northern Italy, clear diction, close microphone, quiet room.",
    "EF2": "A young woman in her twenties with a bright, slightly high voice, native Italian speaker from southern Italy, close microphone, quiet room.",
    "EM2": "A young man in his twenties with a light tenor voice, native Italian speaker from Rome, relaxed delivery, close microphone, quiet room.",
    "EF3": "A woman in her fifties with a low, slightly husky voice, native Italian speaker from Milan, close microphone, quiet room.",
    "EM3": "A man in his sixties with a resonant baritone voice, native Italian speaker from Florence, close microphone, quiet room.",
    # Product voices chosen 2026-11-19 (designs M4 and F3); their neutral data are clones of data/voice-refs/.
    "EM4": "A man in his thirties with a soft, gentle, medium-low voice, native Italian speaker from Tuscany, reassuring and kind, clear diction, close microphone, quiet room.",
    "EF4": "A woman in her forties with a low, smooth, assured voice, native Italian speaker from Milan, a news anchor, clear diction, close microphone, quiet room.",
}
EMOTIONS = {  # label: (slightly, very) style clauses
    "neutral": ("Speaking in a calm, neutral, matter-of-fact tone.",) * 2,
    "joy": ("Speaking in a slightly happy, cheerful tone.", "Speaking very happily and excitedly, full of joy, smiling."),
    "sadness": ("Speaking in a slightly sad, subdued tone.", "Speaking very sadly, close to tears, slow and heavy."),
    "anger": ("Speaking in a slightly irritated, annoyed tone.", "Speaking very angrily, with a raised, tense voice."),
    "fear": ("Speaking in a slightly worried, uneasy tone.", "Speaking in great fear, trembling, breathless and anxious."),
    "surprise": ("Speaking in a slightly surprised tone.", "Speaking with great astonishment, very surprised and amazed."),
    "disgust": ("Speaking with slight distaste.", "Speaking with strong disgust and revulsion."),
}


# Direct route (default): VoiceDesign speaks emotion-congruent sentences (data/emotion-texts.json) with
# English style instructions. Listening test 2026-11-17: English instructions on emotional text give
# convincing emotions; neutral text cloned from emotional references (--mode clone) did not.
STYLES_EN = {
    "neutral": ("Speaking in a calm, neutral, matter-of-fact tone.",) * 2,
    "joy": ("Speaking happily, cheerful and warm, with a smile in the voice.",
            "Bursting with joy, laughing while speaking, very excited and energetic."),
    "sadness": ("Speaking sadly, a little subdued and slow.",
                "Very sad, quietly crying, slow, broken voice with long pauses."),
    "anger": ("Speaking with irritation, annoyed and tense.",
              "Shouting furiously, very angry, loud and tense, fast aggressive delivery."),
    "fear": ("Speaking nervously, worried and uneasy.",
             "Terrified, whispering in panic, trembling, breathless and fast."),
    "surprise": ("Speaking with mild surprise, curious and a little amazed.",
                 "Extremely surprised and astonished, gasping, voice rising with amazement."),
    "disgust": ("Speaking with distaste, slightly repulsed.",
                "Strongly disgusted and repulsed, voice full of revulsion and contempt."),
}


def design_jobs(voices, texts, per_combination):
    jobs = []
    for voice in voices:
        for emotion, pool in texts.items():
            levels = [(0.0, 0)] if emotion == "neutral" else [(0.5, 0), (1.0, 1)]
            for intensity, which in levels:
                offset = int(hashlib.sha256(f"{voice}{emotion}{intensity}".encode()).hexdigest(), 16) % len(pool)
                for k in range(min(per_combination, len(pool))):
                    text = pool[(offset + k) % len(pool)]
                    clip = f"emo-{voice}-{emotion}-{intensity}-{hashlib.sha256(text.encode()).hexdigest()[:10]}"
                    jobs.append({"id": clip, "text": text, "voice_id": voice, "emotion": emotion, "intensity": intensity,
                                 "instruct": f"{VOICES[voice]} {STYLES_EN[emotion][which]}"})
    return jobs


def references_plan(voices, emotions):
    plan = []
    for voice in voices:
        for emotion in emotions:
            levels = [(0.0, 0)] if emotion == "neutral" else [(0.5, 0), (1.0, 1)]
            for intensity, which in levels:
                plan.append({"voice": voice, "emotion": emotion, "intensity": intensity,
                             "instruct": f"{VOICES[voice]} {EMOTIONS[emotion][which]}"})
    return plan


def script_pool(manifests, count, seed, exclude_texts):
    texts = []
    for manifest in manifests:
        for line in Path(manifest).read_text().splitlines():
            text = json.loads(line)["text"].strip()
            words = len(text.split())
            if 7 <= words <= 22 and text not in exclude_texts and not any(ch.isdigit() for ch in text):
                texts.append(text)
    texts = sorted(set(texts))
    random.Random(seed).shuffle(texts)
    return texts[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="2 voices, 5 sentences per reference")
    parser.add_argument("--voices", nargs="*", default=None)
    parser.add_argument("--clips-per-reference", type=int, default=40)
    parser.add_argument("--texts", nargs="+", default=["data/it-multi-annotations/train.jsonl"])
    parser.add_argument("--exclude-texts", nargs="*", default=["data/it-multi-annotations/validation.jsonl", "data/it-multi-annotations/test.jsonl"])
    parser.add_argument("--max-wer", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--mode", choices=("design", "clone"), default="design")
    parser.add_argument("--emotion-texts", type=Path, default=Path("data/emotion-texts.json"))
    args = parser.parse_args()
    voices = args.voices or (["EF1", "EM1"] if args.pilot else list(VOICES))
    per_reference = 5 if args.pilot else args.clips_per_reference
    for sub in ("audio", "references"):
        (args.root / sub).mkdir(parents=True, exist_ok=True)
    if args.mode == "design":
        return design_main(args, voices, per_reference)

    from faster_whisper import WhisperModel
    from jiwer import wer
    from qwen_tts import Qwen3TTSModel
    import qwen_fast
    qwen_fast.patch_graphs()   # sub-code loop as a CUDA graph: ~2.4x throughput, same quality (decision log 6.9)
    from tiny_voca.evaluate import italian_normalizer
    normalize = italian_normalizer
    whisper = WhisperModel("small", device="cuda", compute_type="float16")

    def transcribe(wave, rate):
        import librosa
        audio = librosa.resample(wave.astype(np.float32), orig_sr=rate, target_sr=16000)
        segments, _ = whisper.transcribe(audio, language="it", beam_size=1)
        return " ".join(s.text for s in segments)

    def error(text, wave, rate):
        reference = normalize(text)
        return wer(reference, normalize(transcribe(wave, rate))) if reference.strip() else 1.0

    # 1. Emotional references with VoiceDesign.
    plan = references_plan(voices, EMOTIONS)
    missing = [p for p in plan if not (args.root / "references" / f"{p['voice']}-{p['emotion']}-{p['intensity']}.wav").exists()]
    if missing:
        design = Qwen3TTSModel.from_pretrained(DESIGN, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        for item in missing:
            path = args.root / "references" / f"{item['voice']}-{item['emotion']}-{item['intensity']}.wav"
            best = None
            for attempt in range(3):
                torch.manual_seed(args.seed + attempt)
                waves, rate = design.generate_voice_design(text=REFERENCE_TEXT, instruct=item["instruct"], language="Italian")
                score = error(REFERENCE_TEXT, waves[0], rate)
                if best is None or score < best[0]:
                    best = (score, waves[0], rate)
                if score <= args.max_wer:
                    break
            sf.write(path, best[1], best[2])
            print(json.dumps({"reference": path.name, "wer": round(best[0], 3)}), flush=True)
        del design
        torch.cuda.empty_cache()

    # 2. Bulk cloning with Base, screened by Whisper.
    exclude = {json.loads(l)["text"].strip() for m in args.exclude_texts for l in Path(m).read_text().splitlines()}
    texts = script_pool(args.texts, per_reference * len(plan), args.seed, exclude)
    manifest, rejected = args.root / "manifest.jsonl", args.root / "rejected.jsonl"
    done = {json.loads(l)["id"] for f in (manifest, rejected) if f.exists() for l in f.read_text().splitlines()}
    base = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    start = time.time()
    for r, item in enumerate(plan):
        reference = args.root / "references" / f"{item['voice']}-{item['emotion']}-{item['intensity']}.wav"
        prompt = base.create_voice_clone_prompt(ref_audio=str(reference), ref_text=REFERENCE_TEXT)
        jobs = []
        for k in range(per_reference):
            text = texts[(r * per_reference + k) % len(texts)]
            clip = f"emo-{item['voice']}-{item['emotion']}-{item['intensity']}-{hashlib.sha256(text.encode()).hexdigest()[:10]}"
            if clip not in done:
                jobs.append((clip, text))
        for i in range(0, len(jobs), args.batch):
            batch = jobs[i:i + args.batch]
            for attempt in range(2):
                torch.manual_seed(args.seed + 100 * attempt + i)
                waves, rate = base.generate_voice_clone(text=[t for _, t in batch], language=["Italian"] * len(batch),
                                                        voice_clone_prompt=prompt * len(batch) if isinstance(prompt, list) else prompt)
                retry = []
                for (clip, text), wave in zip(batch, waves):
                    score = error(text, wave, rate)
                    row = {"id": clip, "text": text, "voice_id": item["voice"], "emotion": item["emotion"], "intensity": item["intensity"],
                           "wer": round(score, 3), "audio": f"audio/{clip}.wav", "origin": "qwen3-tts-synthetic",
                           "source_license": "synthetic (Qwen3-TTS, Apache-2.0 weights)"}
                    if score <= args.max_wer:
                        sf.write(args.root / row["audio"], wave, rate)
                        with open(manifest, "a") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    elif attempt == 0:
                        retry.append((clip, text))
                    else:
                        with open(rejected, "a") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                batch = retry
                if not batch:
                    break
        print(json.dumps({"reference": reference.name, "done": r + 1, "of": len(plan), "minutes": round((time.time() - start) / 60, 1)}), flush=True)


def design_main(args, voices, per_combination):
    from faster_whisper import WhisperModel
    from jiwer import wer
    from qwen_tts import Qwen3TTSModel
    import qwen_fast
    qwen_fast.patch_graphs()   # sub-code loop as a CUDA graph: ~2.4x throughput, same quality (decision log 6.9)
    import librosa
    from tiny_voca.evaluate import italian_normalizer as normalize
    whisper = WhisperModel("small", device="cuda", compute_type="float16")
    texts = json.loads(args.emotion_texts.read_text())
    jobs = design_jobs(voices, texts, per_combination)
    manifest, rejected = args.root / "manifest.jsonl", args.root / "rejected.jsonl"
    done = {json.loads(l)["id"] for f in (manifest, rejected) if f.exists() for l in f.read_text().splitlines()}
    jobs = [j for j in jobs if j["id"] not in done]
    print(json.dumps({"to_generate": len(jobs), "already_done": len(done)}), flush=True)
    model = Qwen3TTSModel.from_pretrained(DESIGN, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    start = time.time()
    for i in range(0, len(jobs), args.batch):
        batch = jobs[i:i + args.batch]
        for attempt in range(2):
            torch.manual_seed(args.seed + 1000 * attempt + i)
            waves, rate = model.generate_voice_design(text=[j["text"] for j in batch], instruct=[j["instruct"] for j in batch],
                                                      language=["Italian"] * len(batch))
            retry = []
            for job, wave in zip(batch, waves):
                audio16 = librosa.resample(wave.astype(np.float32), orig_sr=rate, target_sr=16000)
                heard = " ".join(seg.text for seg in whisper.transcribe(audio16, language="it", beam_size=1)[0])
                score = wer(normalize(job["text"]), normalize(heard))
                row = {k: job[k] for k in ("id", "text", "voice_id", "emotion", "intensity", "instruct")}
                row.update({"wer": round(score, 3), "audio": f"audio/{job['id']}.wav", "origin": "qwen3-tts-voicedesign-synthetic",
                            "source_license": "synthetic (Qwen3-TTS VoiceDesign, Apache-2.0 weights)"})
                if score <= args.max_wer:
                    sf.write(args.root / row["audio"], wave, rate)
                    with open(manifest, "a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                elif attempt == 0:
                    retry.append(job)
                else:
                    with open(rejected, "a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            batch = retry
            if not batch:
                break
        print(json.dumps({"done": min(i + args.batch, len(jobs)), "of": len(jobs), "minutes": round((time.time() - start) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
