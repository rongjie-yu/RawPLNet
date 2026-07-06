import unittest

import torch
from torch import nn

from hawp.fsl.adapt_raw import run_adapt_step


class FakePointDetector(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(offset)))

    def forward_dense(self, images):
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


if __name__ == "__main__":
    unittest.main()
