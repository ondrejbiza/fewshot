import argparse

import rospy

from src import utils
from src.real_world import perception
from src.real_world.point_cloud_proxy import PointCloudProxy


def main(args):

    rospy.init_node("perception")
    pc_proxy = PointCloudProxy()

    cloud = pc_proxy.get_all()
    assert cloud is not None

    if args.task == "mug_tree":
        mug_pcd, tree_pcd = perception.mug_tree_segmentation(
            cloud, short_platform=args.short_platform, tall_platform=args.tall_platform
        )

        canon_mug = utils.CanonObj.from_pickle("data/230227_ndf_mugs_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_tree = utils.CanonObj.from_pickle("data/230227_ndf_trees_scale_large_pca_8_dim_alp_0_01.pkl")

        perception.warping(
            mug_pcd, tree_pcd, canon_mug, canon_tree, any_rotation=args.any_rotation
        )
    elif args.task == "bowl_on_mug":
        bowl_pcd, mug_pcd = perception.bowl_mug_segmentation(cloud, platform_1=True)

        canon_bowl = utils.CanonObj.from_pickle("data/230227_ndf_bowls_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_mug = utils.CanonObj.from_pickle("data/230227_ndf_mugs_scale_large_pca_8_dim_alp_0_01.pkl")

        perception.warping(
            bowl_pcd, mug_pcd, canon_bowl, canon_mug, any_rotation=args.any_rotation
        )
    elif args.task == "bottle_in_box":
        bottle_pcd, box_pcd = perception.bottle_box_segmentation(cloud)

        canon_bottle = utils.CanonObj.from_pickle("data/230227_ndf_bottles_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_box = utils.CanonObj.from_pickle("data/230227_boxes_scale_large_pca_8_dim_alp_0_01.pkl")

        perception.warping(
            bottle_pcd, box_pcd, canon_bottle, canon_box, any_rotation=args.any_rotation
        )
    else:
        raise ValueError("Invalid task.")


parser = argparse.ArgumentParser("Find objects for a particular task, create warps.")
parser.add_argument("task", help="[mug_tree, bowl_on_mug, bottle_in_box]")
parser.add_argument("-a", "--any-rotation", default=False, action="store_true")
parser.add_argument("-t", "--tall-platform", default=False, action="store_true")
parser.add_argument("-s", "--short-platform", default=False, action="store_true")
main(parser.parse_args())
