import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.real_world import utils


class TestUtils(unittest.TestCase):

    def test_move_hand_back(self):

        pos = np.array([-1.2, 0.3, 4.3], dtype=np.float32)
        quat = Rotation.from_euler("zyx", (0., 0., 0.)).as_quat()
        delta = 0.1

        out = utils.move_hand_back(pos, quat, delta)
        ref_pos = np.copy(pos)
        ref_pos[2] -= delta

        np.testing.assert_allclose(out[0], ref_pos)
        np.testing.assert_equal(out[1], quat)

    def test_to_pose_message(self):

        pos = np.random.normal(0, 1, 3)
        quat = Rotation.from_euler("zyx", np.random.uniform(0, 2 * np.pi, 3)).as_quat()
        msg = utils.to_pose_message(pos, quat)
        self.assertEqual(msg.position.x, pos[0])
        self.assertEqual(msg.position.y, pos[1])
        self.assertEqual(msg.position.z, pos[2])
        self.assertEqual(msg.orientation.x, quat[0])
        self.assertEqual(msg.orientation.y, quat[1])
        self.assertEqual(msg.orientation.z, quat[2])
        self.assertEqual(msg.orientation.w, quat[3])
