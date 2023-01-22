import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from online_isec import utils


class TestUtils(unittest.TestCase):

    def setUp(self):
        np.random.seed(2023)


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
