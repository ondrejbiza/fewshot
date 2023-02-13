import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from src import viz_utils
from src.object_warping import ObjectWarpingSE2Batch, ObjectSE2Batch, ObjectWarpingSE3Batch, warp_to_pcd_se2, warp_to_pcd_se3
from src.utils import CanonObj


class VizUtilsShow(unittest.TestCase):

    def test_show_pcd_plotly(self):

        mug_pcd = np.load("data/test_data/real_mug_pcd.npy")

        viz_utils.show_pcd_plotly(mug_pcd, center=True)

    def test_show_pcds_pyplot(self):

        tree_pcd = np.load("data/test_data/real_tree_real_pc.npy")
        mug_pcd = np.load("data/test_data/real_mug_pcd.npy")

        viz_utils.show_pcds_pyplot({
            "tree": tree_pcd,
            "mug": mug_pcd
        }, center=True)

    def test_show_pcds_plotly(self):

        tree_pcd = np.load("data/test_data/real_tree_real_pc.npy")
        mug_pcd = np.load("data/test_data/real_mug_pcd.npy")

        viz_utils.show_pcds_plotly({
            "tree": tree_pcd,
            "mug": mug_pcd
        }, center=True)

    def test_show_pcds_pyplot(self):

        tree_pcd = np.load("data/test_data/real_tree_real_pc.npy")
        mug_pcd = np.load("data/test_data/real_mug_pcd.npy")

        viz_utils.show_pcds_pyplot({
            "tree": tree_pcd,
            "mug": mug_pcd
        }, center=True)


    def setUp(self):
        np.random.seed(2023)
