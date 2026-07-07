import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hawp.fsl.dataset.train_dataset import TrainDataset
from hawp.fsl.dataset.transforms import Compose, Resize, ToTensor
from hawp.fsl.raw.synthesis import RawSynthesisConfig


class FakeRawSynthesizer:
    def __init__(self):
        self.seen_rgb = None
        self.iter_indices = []

    def synthesize_rgb(self, rgb, iter_idx=0):
        self.seen_rgb = rgb.copy()
        self.iter_indices.append(iter_idx)
        raw = np.zeros_like(rgb, dtype=np.float32)
        raw[..., 0] = 0.25
        raw[..., 1] = 0.25
        raw[..., 2] = 0.25
        return raw


class TrainDatasetRawTest(unittest.TestCase):
    def _make_tiny_wireframe_dataset(self, root):
        image_root = root / "images"
        image_root.mkdir()
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[..., 0] = 10
        rgb[..., 1] = 20
        rgb[..., 2] = 30
        cv2.imwrite(str(image_root / "sample.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        annotations = [
            {
                "filename": "sample.png",
                "width": 4,
                "height": 4,
                "junctions": [[1.0, 1.0], [2.0, 2.0]],
                "edges_positive": [[0, 1]],
                "edges_negative": [[0, 0]],
            }
        ]
        ann_file = root / "train.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")
        return image_root, ann_file

    def test_train_dataset_synthesizes_raw_after_rgb_read(self):
        import hawp.fsl.dataset.train_dataset as train_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            image_root, ann_file = self._make_tiny_wireframe_dataset(Path(tmpdir))
            fake_synth = FakeRawSynthesizer()
            original_randint = train_dataset.random.randint
            random_args = []

            def fixed_randint(a, b):
                random_args.append((a, b))
                return 5

            train_dataset.random.randint = fixed_randint
            try:
                dataset = TrainDataset(
                    str(image_root),
                    str(ann_file),
                    transform=Compose([Resize(4, 4, 4, 4), ToTensor()]),
                    augmentation=1,
                    raw_config=RawSynthesisConfig(invisp_checkpoint="/tmp/fake.pth"),
                    raw_synthesizer=fake_synth,
                )

                image, ann = dataset[0]
            finally:
                train_dataset.random.randint = original_randint

        self.assertEqual(random_args, [(1, 7)])
        self.assertEqual(fake_synth.seen_rgb.shape, (4, 4, 3))
        np.testing.assert_array_equal(fake_synth.seen_rgb[0, 0], np.array([10, 20, 30]))
        self.assertEqual(tuple(image.shape), (3, 4, 4))
        self.assertAlmostEqual(image[0].min().item(), 0.25)
        self.assertAlmostEqual(image[1].min().item(), 0.25)
        self.assertAlmostEqual(image[2].min().item(), 0.25)
        self.assertEqual(ann["width"], 4)
        self.assertEqual(ann["height"], 4)

    def test_adapt_dataset_synthesizes_raw_before_transform_and_returns_teacher_gray(self):
        import hawp.fsl.dataset.train_dataset as train_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            image_root, ann_file = self._make_tiny_wireframe_dataset(Path(tmpdir))
            fake_synth = FakeRawSynthesizer()
            original_randint = train_dataset.random.randint
            train_dataset.random.randint = lambda a, b: 5
            try:
                dataset = TrainDataset(
                    str(image_root),
                    str(ann_file),
                    transform=Compose([Resize(2, 2, 2, 2), ToTensor()]),
                    augmentation=1,
                    raw_config=RawSynthesisConfig(invisp_checkpoint="/tmp/fake.pth"),
                    raw_synthesizer=fake_synth,
                    return_adapt_pair=True,
                )

                (teacher_gray, student_raw), ann = dataset[0]
            finally:
                train_dataset.random.randint = original_randint

        self.assertEqual(fake_synth.seen_rgb.shape, (4, 4, 3))
        np.testing.assert_array_equal(fake_synth.seen_rgb[0, 0], np.array([10, 20, 30]))
        self.assertEqual(tuple(teacher_gray.shape), (1, 2, 2))
        self.assertEqual(tuple(student_raw.shape), (3, 2, 2))
        self.assertAlmostEqual(float(teacher_gray[0, 0, 0]), (0.299 * 10 + 0.587 * 20 + 0.114 * 30) / 255.0)
        self.assertAlmostEqual(student_raw[0].min().item(), 0.25)
        self.assertEqual(ann["width"], 2)
        self.assertEqual(ann["height"], 2)

    def test_train_dataset_uses_global_raw_step_counter_not_sample_index(self):
        import hawp.fsl.dataset.train_dataset as train_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            image_root, ann_file = self._make_tiny_wireframe_dataset(Path(tmpdir))
            fake_synth = FakeRawSynthesizer()
            original_randint = train_dataset.random.randint
            train_dataset.random.randint = lambda a, b: 5
            try:
                dataset = TrainDataset(
                    str(image_root),
                    str(ann_file),
                    transform=Compose([Resize(4, 4, 4, 4), ToTensor()]),
                    augmentation=1,
                    raw_config=RawSynthesisConfig(invisp_checkpoint="/tmp/fake.pth"),
                    raw_synthesizer=fake_synth,
                )

                dataset[0]
                dataset[0]
            finally:
                train_dataset.random.randint = original_randint

        self.assertEqual(fake_synth.iter_indices, [0, 1])

    def test_removed_rgb_domain_noise_functions_are_not_exported(self):
        import hawp.fsl.dataset.train_dataset as train_dataset

        self.assertFalse(hasattr(train_dataset, "additive_gaussian_noise"))
        self.assertFalse(hasattr(train_dataset, "additive_speckle_noise"))


if __name__ == "__main__":
    unittest.main()
