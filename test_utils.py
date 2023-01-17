import pickle
from unittest import TestCase
import numpy as np
from scipy.spatial.transform import Rotation
import torch

import utils


class TestUtils(TestCase):

    def setUp(self):

        np.random.seed(2023)

    def test_canon_to_pc(self):

        with open("data/mugs_pca.pkl", "rb") as f:
            canon = pickle.load(f)
        latent = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        center = np.array([0.5, -0.2, 0.3], dtype=np.float32)
        angle = np.array([np.pi / 3.], dtype=np.float32)

        pc = utils.canon_to_pc(canon, (latent, center, angle))
        self.assertEqual(pc.shape, canon["canonical_obj"].shape)

    def test_canon_to_transformed_pc(self):

        with open("data/mugs_pca.pkl", "rb") as f:
            canon = pickle.load(f)
        latent = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        center = np.array([0.5, -0.2, 0.3], dtype=np.float32)
        angle = np.array([np.pi / 3.], dtype=np.float32)

        pc = utils.canon_to_pc(canon, (latent, center, angle))
        pc_T = utils.canon_to_transformed_pc(canon, (latent, center, angle))
        
        T = utils.pos_rot_to_transform(center, utils.yaw_to_rot(angle))
        tmp = utils.transform_pointcloud_2(pc_T, np.linalg.inv(T))

        np.testing.assert_almost_equal(pc, tmp)

    def test_compute_relative_transform(self):

        T_from = utils.pos_quat_to_transform(
            np.random.normal(3, 1, size=3),
            Rotation.from_euler("zyx", np.random.uniform(0, 2 * np.pi, size=3)).as_quat()
        )
        T_to = utils.pos_quat_to_transform(
            np.random.normal(3, 1, size=3),
            Rotation.from_euler("zyx", np.random.uniform(0, 2 * np.pi, size=3)).as_quat()
        )

        T = utils.compute_relative_transform(T_from, T_to)
        tmp = np.matmul(T_from, T)
        np.testing.assert_almost_equal(tmp, T_to, decimal=4)

    def test_move_hand_back(self):

        pos = np.array([-1.2, 0.3, 4.3], dtype=np.float32)
        quat = Rotation.from_euler("zyx", (0., 0., 0.)).as_quat()
        delta = 0.1

        out = utils.move_hand_back((pos, quat), delta)
        ref_pos = np.copy(pos)
        ref_pos[2] -= delta

        np.testing.assert_allclose(out[0], ref_pos)
        np.testing.assert_equal(out[1], quat)

    def test_yaw_to_rot_numpy_and_pt(self):

        # TODO: what about the unit dimensions?
        yaws = np.random.uniform(0, 2 * np.pi, (100, 1)).astype(np.float32)

        for yaw in yaws:
            yaw_pt = torch.tensor(yaw, dtype=torch.float32, device="cpu")
            rot = utils.yaw_to_rot(yaw)
            rot_pt = utils.yaw_to_rot_pt(yaw_pt).numpy()
            np.testing.assert_almost_equal(rot, rot_pt)
