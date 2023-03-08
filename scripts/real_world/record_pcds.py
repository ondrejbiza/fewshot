import argparse
import os
import pickle

import rospy

from src import viz_utils
from src.real_world import perception
from src.real_world.point_cloud_proxy import PointCloudProxy


def main(args):

    rospy.init_node("perception")
    pc_proxy = PointCloudProxy()

    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    i = 0
    while True:
        cloud = pc_proxy.get_all()
        assert cloud is not None

        if args.task == "mug_tree":
            mug_pc, tree_pc = perception.mug_tree_segmentation(cloud)
            d = {"mug": mug_pc, "tree": tree_pc}
        elif args.task == "bowl_on_mug":
            bowl_pcd, mug_pcd = perception.bowl_mug_segmentation(cloud, platform_1=True)
            d = {"bowl": bowl_pcd, "mug": mug_pcd}
        elif args.task == "bottle_in_box":
            bottle_pcd, box_pcd = perception.bottle_box_segmentation(cloud)
            d = {"bottle": bottle_pcd, "box": box_pcd}
        else:
            raise ValueError("Invalid task.")

        save_file = os.path.join(args.save_folder, f"{args.task}_{i}.pkl")
        with open(save_file, "wb") as f:
            pickle.dump(d, f)

        viz_utils.show_pcds_plotly(d)
        i += 1
        input("Next")


parser = argparse.ArgumentParser("Find objects for a particular task.")
parser.add_argument("task", help="[mug_tree, bowl_on_mug, bottle_in_box]")
parser.add_argument("save_folder")
main(parser.parse_args())
