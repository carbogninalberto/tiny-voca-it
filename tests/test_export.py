import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tiny_voca.config import Config
from tiny_voca.export import Exportable, export
from tiny_voca.losses import VocosMel
from tiny_voca.model import TinyVoca
from tiny_voca.vocoder import Vocoder, VocoderConfig
from tiny_voca.text import PHONE_IDS


def small_model(jitter=0.0, postfilter=0, head=0, gate=1.0):
    torch.manual_seed(0)
    model = TinyVoca(Config(excitation_jitter=jitter, postfilter_channels=postfilter, head_channels=head, unvoiced_pulse_gain=gate)).eval()
    if postfilter:   # a trained post-filter is not the identity
        torch.nn.init.normal_(model.vocoder.postfilter.output.weight, std=0.05)
    if head:
        torch.nn.init.normal_(model.vocoder.head.output.weight, std=0.02)
    # A fresh model predicts arbitrary durations; pin them to a few frames per token.
    torch.nn.init.zeros_(model.duration.out.weight)
    model.duration.out.bias.data[:] = torch.tensor([1.5, 1.0])
    torch.nn.init.normal_(model.vocoder.harmonic.weight, std=0.05)
    torch.nn.init.normal_(model.vocoder.noise.weight, std=0.05)
    return model


def utterance(length=12):
    phones = sorted(PHONE_IDS)
    tokens = torch.tensor([[phones[(7 * i) % len(phones)] for i in range(length)]])
    return tokens, (torch.arange(length)[None] % 3)


class ExportTest(unittest.TestCase):
    def test_rewrite_matches_synthesize(self):
        for jitter, postfilter, head, gate in ((0.0, 0, 0, 1.0), (0.04, 0, 0, 1.0), (0.0, 8, 0, 1.0), (0.0, 0, 16, 1.0), (0.0, 0, 0, 0.5)):
            model = small_model(jitter, postfilter, head, gate)
            tokens, stress = utterance()
            for voice in (0, 1):
                reference, lengths, details = model.synthesize(tokens, stress, torch.tensor([tokens.shape[1]]), torch.tensor([voice]),
                                                               speed=1.25, pitch_shift=0.1, pitch_range=1.3)
                noise = torch.randn(int(lengths[0]), generator=torch.Generator().manual_seed(0))[None]
                one = lambda value: torch.tensor([value])
                with torch.no_grad():
                    audio, durations = Exportable(model, external_noise=True)(tokens, stress, torch.tensor([voice]), one(1.25), one(0.1), one(1.3), noise)
                self.assertTrue(torch.equal(durations, details["durations"]))
                self.assertEqual(audio.shape, reference.shape)
                self.assertLess(float((audio - reference).abs().max()), 1e-2 * float(reference.abs().max()))

    def test_onnx_runs_at_other_lengths(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime is not installed")
        for jitter, postfilter, head, gate in ((0.0, 0, 0, 1.0), (0.04, 0, 0, 1.0), (0.0, 8, 0, 1.0), (0.0, 0, 16, 1.0), (0.0, 0, 0, 0.5)):
            model = small_model(jitter, postfilter, head, gate)
            with tempfile.TemporaryDirectory() as directory:
                path = export(model, Path(directory) / "model.onnx", external_noise=True)
                session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                for length in (5, 40):
                    tokens, stress = utterance(length)
                    reference, lengths, _ = model.synthesize(tokens, stress, torch.tensor([length]), torch.tensor([1]))
                    noise = np.zeros((1, model.config.hop * model.config.max_frames), dtype=np.float32)
                    noise[0, :int(lengths[0])] = torch.randn(int(lengths[0]), generator=torch.Generator().manual_seed(0)).numpy()
                    one = lambda value: np.array([value], dtype=np.float32)
                    audio, _ = session.run(None, {"tokens": tokens.numpy(), "stress": stress.numpy(), "voice": np.array([1]),
                                                  "speed": one(1.0), "pitch_shift": one(0.0), "pitch_range": one(1.0), "noise": noise})
                    self.assertEqual(audio.shape, tuple(reference.shape))
                    self.assertLess(float(np.abs(audio - reference.numpy()).max()), 2e-2 * float(reference.abs().max()))


def small_vocoder(high_hz=0.0):
    torch.manual_seed(1)
    vocoder = Vocoder(VocoderConfig(channels=32, intermediate=64, layers=2, prior=True, prior_noise=0.0,
                                    prior_high_hz=high_hz, prior_high_gain=0.1)).eval()
    torch.nn.init.normal_(vocoder.prior_in.weight, std=0.05)
    return vocoder


class NeuralExportTest(unittest.TestCase):
    def test_neural_vocoder_rewrite_matches_vocoder(self):
        for high_hz in (0.0, 2500.0):
            model, vocoder = small_model(), small_vocoder(high_hz)
            wrapper = Exportable(model, vocoder=vocoder).eval()
            frames = 37
            wave = 0.1 * torch.randn(1, frames * model.config.hop, generator=torch.Generator().manual_seed(2))
            f0 = 120 + 30 * torch.rand(1, frames, generator=torch.Generator().manual_seed(3))
            voiced = (torch.arange(frames) % 9 < 6).float()[None]
            with torch.no_grad():
                reference = vocoder(VocosMel()(wave).transpose(1, 2), f0, voiced)
                audio = wrapper._neural(wave, f0, voiced, torch.tensor(frames))
            self.assertEqual(audio.shape, reference.shape)
            self.assertLess(float((audio - reference).abs().max()), 1e-3 * float(reference.abs().max()))

    def test_onnx_with_vocoder_and_emotion_matches_pytorch(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime is not installed")
        torch.manual_seed(0)
        model = TinyVoca(Config(emotions=("neutral", "happy"))).eval()
        torch.nn.init.zeros_(model.duration.out.weight)
        model.duration.out.bias.data[:] = torch.tensor([1.5, 1.0])
        torch.nn.init.normal_(model.emotions.weight, std=0.5)
        vocoder = small_vocoder()
        with tempfile.TemporaryDirectory() as directory:
            path = export(model, Path(directory) / "model.onnx", external_noise=True, vocoder=vocoder)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.assertEqual([i.name for i in session.get_inputs()][6:8], ["emotion", "intensity"])
            tokens, stress = utterance(20)
            noise = torch.randn(1, model.config.hop * model.config.max_frames, generator=torch.Generator().manual_seed(0))
            one = lambda value: torch.tensor([value], dtype=torch.float32)
            with torch.no_grad():
                expected, _ = Exportable(model, True, vocoder)(tokens, stress, torch.tensor([0]), one(1.0), one(0.0), one(1.0),
                                                               torch.tensor([1]), one(0.8), noise)
            audio, _ = session.run(None, {"tokens": tokens.numpy(), "stress": stress.numpy(), "voice": np.array([0]),
                                          "speed": one(1.0).numpy(), "pitch_shift": one(0.0).numpy(), "pitch_range": one(1.0).numpy(),
                                          "emotion": np.array([1]), "intensity": one(0.8).numpy(), "noise": noise.numpy()})
            self.assertEqual(audio.shape, tuple(expected.shape))
            self.assertLess(float(np.abs(audio - expected.numpy()).max()), 2e-2 * float(expected.abs().max()))


if __name__ == "__main__":
    unittest.main()
