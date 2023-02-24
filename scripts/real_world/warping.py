import argparse
import time

import numpy as np
from numpy.typing import NDArray
import rospy
import torch

from src import object_warping, utils, viz_utils
from src.real_world import perception
from src.real_world.point_cloud_proxy_sync import PointCloudProxy


def main(args):

    rospy.init_node("perception")
    pc_proxy = PointCloudProxy()

    cloud = pc_proxy.get_all()
    assert cloud is not None

    if args.task == "mug_tree":
        mug_pcd, tree_pcd = perception.mug_tree_simple_perception(cloud)

        canon_mug = utils.CanonObj.from_pickle("data/230213_ndf_mugs_scale_large_pca_8_dim_alp_0.01.pkl")
        canon_tree = utils.CanonObj.from_pickle("data/230213_ndf_trees_scale_large_pca_8_dim_alp_2.pkl")
        
        warp = object_warping.ObjectWarpingSE2Batch(
            canon_mug, mug_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=1000, object_size_reg=0.1, scaling=True, init_scale=0.1)
        mug_pcd_complete, _, _ = object_warping.warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

        warp = object_warping.ObjectWarpingSE2Batch(
            canon_tree, tree_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=1000, object_size_reg=0.1, scaling=True, init_scale=0.1)
        tree_pcd_complete, _, _ = object_warping.warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

        d = {
            "mug": mug_pcd,
            "tree": tree_pcd,
            "complete_mug": mug_pcd_complete,
            "complete_tree": tree_pcd_complete}
    else:
        raise ValueError("Invalid task.")

    viz_utils.show_pcds_plotly(d)


parser = argparse.ArgumentParser("Find objects for a particular task, create warps.")
parser.add_argument("task", help="[mug_tree, bowl_on_mug, bottle_in_container]")
main(parser.parse_args())
