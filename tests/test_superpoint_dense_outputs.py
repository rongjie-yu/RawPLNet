import unittest

import torch

from hawp.fsl.backbones.point_line import SuperPoint


class SuperPointDenseOutputTest(unittest.TestCase):
    def test_superpoint_forward_dense_exposes_detector_logits_and_descriptors(self):
        model = SuperPoint({})
        image = torch.zeros(1, 1, 64, 64)

        outputs = model.forward_dense(image)

        self.assertEqual(outputs["detector_logits"].shape, (1, 65, 8, 8))
        self.assertEqual(outputs["detector_log_probs"].shape, (1, 65, 8, 8))
        self.assertEqual(outputs["dense_descriptors"].shape, (1, 256, 8, 8))
        norms = outputs["dense_descriptors"].norm(p=2, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
