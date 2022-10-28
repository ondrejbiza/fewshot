from unittest import TestCase
import numpy as np
from scipy.spatial.transform import Rotation
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

    def test_move_hand_back(self):

        pos = np.array([-1.2, 0.3, 4.3], dtype=np.float32)
        quat = Rotation.from_euler("zyx", (0., 0., 0.)).as_quat()
        delta = 0.1

        out = utils.move_hand_back((pos, quat), delta)
        ref_pos = np.copy(pos)
        ref_pos[2] -= delta

        np.testing.assert_allclose(out[0], ref_pos)
        np.testing.assert_equal(out[1], quat)
