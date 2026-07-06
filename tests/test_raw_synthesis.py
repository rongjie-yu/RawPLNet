import numpy as np
import unittest

from hawp.fsl.raw.synthesis import RawSynthesisConfig, RawSynthesizer


class FakeSimulator:
    def __init__(self):
        self.noise_ratio = None

    def rgb2raw(self, rgb, batched=False):
        assert batched is False
        assert rgb.shape == (8, 10, 3)
        assert rgb.dtype == np.float32
        assert 0.0 <= rgb.min() <= rgb.max() <= 1.0
        return np.ones(rgb.shape[:2], dtype=np.float32) * 2048.0

    def raw2noisyRaw(self, raw, ratio_dec=1, batched=False):
        assert batched is False
        self.noise_ratio = ratio_dec
        return raw + ratio_dec

    def raw2demosaicRaw(self, raw, batched=False):
        assert batched is False
        return np.repeat(raw[..., None], 3, axis=2) / 4096.0


class RawSynthesizerTest(unittest.TestCase):
    def test_synthesizer_returns_gray_replicated_first_channel(self):
        rgb = np.zeros((8, 10, 3), dtype=np.uint8)
        simulator = FakeSimulator()
        synth = RawSynthesizer(
            RawSynthesisConfig(
                invisp_checkpoint="/tmp/fake.pth",
                enable_eld_noise=True,
                noise_maxstep=10,
                output_mode="gray_replicated",
            ),
            simulator=simulator,
        )

        raw = synth.synthesize_rgb(rgb, iter_idx=5)

        self.assertEqual(simulator.noise_ratio, 0.5)
        self.assertEqual(raw.shape, (8, 10, 3))
        self.assertEqual(raw.dtype, np.float32)
        self.assertTrue(np.isfinite(raw).all())
        self.assertGreaterEqual(raw.min(), 0.0)
        self.assertLessEqual(raw.max(), 1.0)
        np.testing.assert_allclose(raw[..., 0], raw[..., 1])
        np.testing.assert_allclose(raw[..., 0], raw[..., 2])

    def test_synthesizer_rejects_missing_checkpoint_without_injected_simulator(self):
        cfg = RawSynthesisConfig(invisp_checkpoint="/tmp/does-not-exist.pth")

        with self.assertRaises(FileNotFoundError):
            RawSynthesizer(cfg)

    def test_synthesizer_rejects_rgb_fallback_mode(self):
        with self.assertRaises(ValueError):
            RawSynthesisConfig(
                invisp_checkpoint="/tmp/fake.pth",
                output_mode="rgb",
            )


if __name__ == "__main__":
    unittest.main()
