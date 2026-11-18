import unittest

import torch

from tiny_voca.losses import VocosMel
from tiny_voca.train_vocoder import complex_distance, learning_rate
from tiny_voca.vocoder import Vocoder, VocoderConfig


class VocoderTest(unittest.TestCase):
    def test_shapes_size_and_one_training_step(self):
        torch.manual_seed(0)
        config = VocoderConfig()
        vocoder = Vocoder(config)
        self.assertLess(sum(p.numel() for p in vocoder.parameters()), 2_500_000)
        features = VocosMel()
        wave = 0.1 * torch.randn(2, 64 * config.hop)
        mel = features(wave)                                  # [B, frames + 1, bands]
        self.assertEqual(mel.shape, (2, 65, config.bands))
        fake = vocoder(mel.transpose(1, 2))
        self.assertEqual(fake.shape, wave.shape)
        optimizer = torch.optim.AdamW(vocoder.parameters(), lr=1e-3)
        before = float((features(fake) - mel).abs().mean())
        for _ in range(5):
            loss = (features(vocoder(mel.transpose(1, 2))) - mel).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss), before)

    def test_complex_distance_sees_phase_not_only_magnitude(self):
        torch.manual_seed(0)
        time = torch.arange(24000) / 24000
        a = torch.sin(2 * torch.pi * 220 * time)[None]
        shifted = torch.sin(2 * torch.pi * 220 * time + 1.5)[None]      # same magnitude spectrum, other phase
        self.assertEqual(float(complex_distance(a, a)), 0.0)
        self.assertGreater(float(complex_distance(a, shifted)), 0.05)
        magnitude = lambda x: torch.stft(x, 1024, 256, window=torch.hann_window(1024), return_complex=True).abs()
        self.assertLess(float((magnitude(a) - magnitude(shifted)).abs().mean()), 0.05 * float(magnitude(a).mean()))

    def test_schedule(self):
        self.assertAlmostEqual(learning_rate(0, 1.0, 10, 100), 0.1)
        self.assertAlmostEqual(learning_rate(10, 1.0, 10, 100), 1.0)
        self.assertAlmostEqual(learning_rate(100, 1.0, 10, 100), 0.0)


if __name__ == "__main__":
    unittest.main()
