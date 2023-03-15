import argparse
import time

import rospy

from src import utils
from src.real_world import perception
from src.real_world.point_cloud_proxy import PointCloudProxy


def main(args):

    rospy.init_node("perception")
    pc_proxy = PointCloudProxy()
    time.sleep(2)

    cloud = pc_proxy.get_all()

    if args.task == "mug_tree":
        mug_pcd, tree_pcd = perception.mug_tree_segmentation(cloud)

        canon_mug = utils.CanonObj.from_pickle("data/230227_ndf_mugs_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_tree = utils.CanonObj.from_pickle("data/230227_ndf_trees_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_mug.init_scale = 0.7
        canon_tree.init_scale = 1.

        perception.warping(
            mug_pcd, tree_pcd, canon_mug, canon_tree, source_any_rotation=args.any_rotation
        )
    elif args.task == "bowl_on_mug":
        bowl_pcd, mug_pcd = perception.bowl_mug_segmentation(cloud)

        canon_bowl = utils.CanonObj.from_pickle("data/230227_ndf_bowls_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_mug = utils.CanonObj.from_pickle("data/230227_ndf_mugs_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_bowl.init_scale = 0.8
        canon_mug.init_scale = 0.7

        perception.warping(
            bowl_pcd, mug_pcd, canon_bowl, canon_mug, source_any_rotation=args.any_rotation
        )
    elif args.task == "bottle_in_box":
        bottle_pcd, box_pcd = perception.bottle_box_segmentation(cloud)

        canon_bottle = utils.CanonObj.from_pickle("data/230227_ndf_bottles_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_box = utils.CanonObj.from_pickle("data/230227_boxes_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_bottle.init_scale = 1.
        canon_box.init_scale = 1.

        perception.warping(
            bottle_pcd, box_pcd, canon_bottle, canon_box, source_any_rotation=args.any_rotation
        )
    else:
        raise ValueError("Invalid task.")


parser = argparse.ArgumentParser("Find objects for a particular task, create warps.")
parser.add_argument("task", help="[mug_tree, bowl_on_mug, bottle_in_box]")
parser.add_argument("-a", "--any-rotation", default=False, action="store_true")
main(parser.parse_args())
