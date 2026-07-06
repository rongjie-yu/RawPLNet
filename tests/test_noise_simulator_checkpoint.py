import unittest
from unittest import mock

import numpy as np
import torch

from hawp.fsl.raw import noise_simulator


class FakeInvISPNet:
    def __init__(self, *args, **kwargs):
        self.loaded = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def load_state_dict(self, state, strict=False):
        self.loaded = (state, strict)


class NoiseSimulatorCheckpointTest(unittest.TestCase):
    def test_checkpoint_load_uses_target_device_map_location(self):
        with mock.patch.object(noise_simulator, "InvISPNet", FakeInvISPNet):
            with mock.patch.object(noise_simulator.torch, "load", return_value={}) as load:
                noise_simulator.NoiseSimulator(device="cpu", ckpt_path="/tmp/fake-canon.pth")

        load.assert_called_once_with("/tmp/fake-canon.pth", map_location="cpu")

    def test_rgb2raw_accepts_numpy_rgb(self):
        simulator = noise_simulator.NoiseSimulator.__new__(noise_simulator.NoiseSimulator)
        simulator.device = "cpu"
        simulator.wb = np.array([2020.0, 1024.0, 1458.0, 1024.0])
        simulator.camera_params = noise_simulator.camera_params

        class IdentityNet:
            def __call__(self, rgb, rev=False):
                self.input_type = type(rgb)
                return rgb

        net = IdentityNet()
        simulator.net = net

        raw = simulator.rgb2raw(np.zeros((4, 4, 3), dtype=np.float32), batched=False)

        self.assertIs(net.input_type, torch.Tensor)
        self.assertEqual(raw.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
