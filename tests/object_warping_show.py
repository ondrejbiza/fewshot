import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from src import viz_utils
from src.object_warping import ObjectWarpingSE2Batch, ObjectSE2Batch, ObjectWarpingSE3Batch, warp_to_pcd_se2, warp_to_pcd_se3
from src.utils import CanonObj


class ObjectWarpingRegressionTest(unittest.TestCase):

    def test_se2_tree_pose(self):

        print("Test SE(2) tree pose:")
        pcd = np.load("data/test_data/real_tree_real_pc.npy")
        canon_tree = CanonObj.from_pickle("data/test_data/real_tree_pc.pkl")

        viz_utils.show_pcd_pyplot(pcd, center=True)

        warp = ObjectSE2Batch(
            canon_tree, pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=None)
        new_pcd, _, _ = warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

        viz_utils.show_pcds_pyplot({
            "pcd": pcd,
            "warp": new_pcd
        }, center=True)

    def test_se2_mug_warping(self):

        print("Test SE(2) mug warping:")
        pcd = np.load("data/test_data/real_mug_pcd.npy")
        canon_mug = CanonObj.from_pickle("data/test_data/ndf_mugs_pca_8_dim.npy")

        viz_utils.show_pcd_pyplot(pcd, center=True)

        warp = ObjectWarpingSE2Batch(
            canon_mug, pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=None, object_size_reg=0.1)
        new_pcd, _, _ = warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

        viz_utils.show_pcds_pyplot({
            "pcd": pcd,
            "warp": new_pcd
        }, center=True)

    def test_se3_mug_warping(self):

        print("Test SE(3) mug warping:")

        pcd = np.load("data/test_data/real_mug_pcd.npy")
        rot = Rotation.random().as_matrix()
        pcd = np.matmul(pcd, rot.T)

        canon_mug = CanonObj.from_pickle("data/test_data/ndf_mugs_pca_8_dim.npy")
        viz_utils.show_pcd_pyplot(pcd, center=True)

        warp = ObjectWarpingSE3Batch(
            canon_mug, pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=1000, object_size_reg=0.1)
        new_pcd, _, _ = warp_to_pcd_se3(warp, n_angles=50, n_batches=3)

        viz_utils.show_pcds_pyplot({
            "pcd": pcd,
            "warp": new_pcd
        }, center=True)

    def setUp(self):
        np.random.seed(2023)
