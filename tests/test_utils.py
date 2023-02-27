import pickle
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src import utils


class TestUtils(unittest.TestCase):

    TEST_MUGS_PCA = "data/test_data/ndf_mugs_pca_8_dim.npy"

    def test_pos_quat_to_transform_and_back(self):
        
        pos = np.random.normal(0, 1, 3).astype(np.float64)
        quat = Rotation.from_euler("zyx", np.random.uniform(-np.pi, np.pi, 3)).as_quat().astype(np.float64)

        pos_, quat_ = utils.transform_to_pos_quat(utils.pos_quat_to_transform(pos, quat))
        
        np.testing.assert_almost_equal(pos_, pos)
        # There are two unique unit quaternions per rotation matrix.
        self.assertTrue(np.allclose(quat_, quat) or np.allclose(-quat_, quat))

    def test_transform_to_pos_quat_and_back(self):
        
        trans = np.eye(4).astype(np.float64)
        trans[:3, :3] = Rotation.random().as_matrix()
        trans[:3, 3] = np.random.normal(0, 1, 3)

        trans_ = utils.pos_quat_to_transform(*utils.transform_to_pos_quat(trans))

        np.testing.assert_almost_equal(trans_, trans)

    def test_canon_to_pc(self):

        with open(self.TEST_MUGS_PCA, "rb") as f:
            canon_d = pickle.load(f)

        latent = np.array([0.1, 0.2, 0.3, 0.4, 0., 0., 0., 0.], dtype=np.float32)
        center = np.array([0.5, -0.2, 0.3], dtype=np.float32)
        quat = Rotation.from_euler("z", np.pi / 3).as_quat()

        canon = utils.CanonObj.from_pickle(self.TEST_MUGS_PCA)
        param = utils.ObjParam(center, quat, latent)

        pcd = canon.to_pcd(param)
        self.assertEqual(pcd.shape, canon_d["canonical_obj"].shape)

    def test_canon_to_transformed_pc(self):

        with open(self.TEST_MUGS_PCA, "rb") as f:
            canon_d = pickle.load(f)

        latent = np.array([0.1, 0.2, 0.3, 0.4, 0., 0., 0., 0.], dtype=np.float32)
        center = np.array([0.5, -0.2, 0.3], dtype=np.float32)
        quat = Rotation.from_euler("z", np.pi / 3).as_quat()

        canon = utils.CanonObj.from_pickle(self.TEST_MUGS_PCA)
        param = utils.ObjParam(center, quat, latent)

        pcd = canon.to_pcd(param)
        pcd_T = canon.to_transformed_pcd(param)
        
        T = utils.pos_quat_to_transform(center, quat)
        tmp = utils.transform_pcd(pcd_T, np.linalg.inv(T))

        np.testing.assert_almost_equal(pcd, tmp)
