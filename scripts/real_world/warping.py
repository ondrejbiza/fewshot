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
        mug_pcd_complete, mug_param, tree_pcd_complete, tree_param, canon_mug, canon_tree, mug_pcd, tree_pcd = perception.mug_tree_perception(
            cloud, any_rotation=args.any_rotation, short_mug_platform=args.short_platform, tall_mug_platform=args.tall_platform
        )

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
parser.add_argument("-a", "--any-rotation", default=False, action="store_true")
parser.add_argument("-t", "--tall-platform", default=False, action="store_true")
parser.add_argument("-s", "--short-platform", default=False, action="store_true")
main(parser.parse_args())
