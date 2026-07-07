import unittest

import torch
from torch import nn

from hawp.fsl.adapt_raw import (
    build_adapt_lr_scheduler,
    maybe_save_noise_warmup_checkpoint,
    run_adapt_step,
)


class FakePointDetector(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(offset)))
        self.seen_images = None

    def forward_dense(self, images):
        self.seen_images = images.detach().clone()
        batch = images.shape[0]
        logits = torch.zeros(batch, 65, 2, 2, device=images.device)
        logits[:, 1] = self.bias + images[:, :1, ::4, ::4].mean(dim=1)
        descriptors = torch.zeros(batch, 4, 2, 2, device=images.device)
        descriptors[:, 0] = self.bias + 1.0
        descriptors[:, 1] = images[:, 0, ::4, ::4]
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        return {
            "detector_logits": logits,
            "detector_log_probs": torch.nn.functional.log_softmax(logits, dim=1),
            "dense_descriptors": descriptors,
        }


class FakeModel(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.point_detector = FakePointDetector(offset)


class AdaptRawStepTest(unittest.TestCase):
    def test_run_adapt_step_updates_student_and_not_teacher(self):
        teacher = FakeModel(offset=0.5)
        student = FakeModel(offset=-0.5)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        rgb = torch.ones(2, 3, 8, 8)
        raw = torch.zeros(2, 3, 8, 8)
        teacher_before = teacher.point_detector.bias.detach().clone()
        student_before = student.point_detector.bias.detach().clone()

        losses = run_adapt_step(teacher, student, rgb, raw, optimizer)

        self.assertEqual(set(losses), {"loss_detector", "loss_descriptor", "loss_total"})
        torch.testing.assert_close(teacher.point_detector.bias, teacher_before)
        self.assertFalse(torch.equal(student.point_detector.bias.detach(), student_before))

    def test_run_adapt_step_feeds_teacher_explicit_grayscale_not_red_channel(self):
        teacher = FakeModel(offset=0.5)
        student = FakeModel(offset=-0.5)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        rgb = torch.zeros(1, 3, 8, 8)
        rgb[:, 0] = 1.0
        raw = torch.zeros(1, 3, 8, 8)

        run_adapt_step(teacher, student, rgb, raw, optimizer)

        expected_gray = torch.full((1, 1, 8, 8), 0.299)
        torch.testing.assert_close(teacher.point_detector.seen_images, expected_gray)

    def test_adapt_cosine_scheduler_holds_lr_until_noise_warmup_finishes(self):
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        scheduler = build_adapt_lr_scheduler(
            optimizer,
            scheduler_name="cosine_after_noise",
            total_steps=10,
            noise_warmup_steps=4,
            min_lr=1e-5,
        )

        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        self.assertEqual(lrs[:4], [1e-4, 1e-4, 1e-4, 1e-4])
        self.assertLess(lrs[4], lrs[3])
        self.assertLess(lrs[-1], 2e-5)
        self.assertGreaterEqual(lrs[-1], 1e-5)

    def test_adapt_lr_scheduler_can_be_disabled(self):
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        scheduler = build_adapt_lr_scheduler(
            optimizer,
            scheduler_name="none",
            total_steps=10,
            noise_warmup_steps=4,
            min_lr=1e-5,
        )

        for _ in range(10):
            optimizer.step()
            scheduler.step()

        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)

    def test_noise_warmup_checkpoint_is_saved_once_when_step_crosses_boundary(self):
        class FakeCheckpointer:
            def __init__(self):
                self.saved = []

            def save(self, name):
                self.saved.append(name)

        checkpointer = FakeCheckpointer()

        saved = maybe_save_noise_warmup_checkpoint(
            checkpointer,
            was_saved=False,
            previous_step=3,
            current_step=4,
            noise_warmup_steps=4,
        )
        saved = maybe_save_noise_warmup_checkpoint(
            checkpointer,
            was_saved=saved,
            previous_step=4,
            current_step=5,
            noise_warmup_steps=4,
        )

        self.assertTrue(saved)
        self.assertEqual(checkpointer.saved, ["adapt_raw_noise_warmup_done"])


if __name__ == "__main__":
    unittest.main()
