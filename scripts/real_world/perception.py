import argparse
import time

import numpy as np
from numpy.typing import NDArray
import rospy

from src import viz_utils
from src.real_world import perception
from src.real_world.point_cloud_proxy_sync import PointCloudProxy


def main(args):

    rospy.init_node("perception")
    pc_proxy = PointCloudProxy()

    cloud = pc_proxy.get_all()
    assert cloud is not None

    if args.task == "mug_tree":
        mug_pc, tree_pc = perception.mug_tree_simple_perception(cloud)
        d = {"mug": mug_pc, "tree": tree_pc}
    else:
        raise ValueError("Invalid task.")

    viz_utils.show_pcds_plotly(d)


parser = argparse.ArgumentParser("Find objects for a particular task.")
parser.add_argument("task", help="[mug_tree, bowl_on_mug, bottle_in_container]")
main(parser.parse_args())
