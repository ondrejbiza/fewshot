from unittest import TestCase
import numpy as np
import torch

import utils


class TestUtils(TestCase):

    def test_yaw_to_rot_numpy_and_pt(self):

        # TODO: what about the unit dimensions?
        yaws = np.random.uniform(0, 2 * np.pi, (100, 1)).astype(np.float32)

        for yaw in yaws:
            yaw_pt = torch.tensor(yaw, dtype=torch.float32, device="cpu")
            rot = utils.yaw_to_rot(yaw)
            rot_pt = utils.yaw_to_rot_pt(yaw_pt).numpy()
            np.testing.assert_almost_equal(rot, rot_pt)
