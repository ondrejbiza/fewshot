import argparse
import time

import rospy

from src import utils
from src.real_world import constants, perception
from src.real_world.point_cloud_proxy import PointCloudProxy


def main(args):

    rospy.init_node("perception")
    pc_proxy = PointCloudProxy()
    time.sleep(2)

    cloud = pc_proxy.get_all()

    if args.task == "mug_tree":
        mug_pcd, tree_pcd = perception.mug_tree_segmentation(cloud)

        canon_mug = utils.CanonObj.from_pickle(constants.NDF_MUGS_PCA_PATH)
        canon_tree = utils.CanonObj.from_pickle(constants.SIMPLE_TREES_PCA_PATH)
        canon_mug.init_scale = constants.NDF_MUGS_INIT_SCALE
        canon_tree.init_scale = constants.SIMPLE_TREES_INIT_SCALE

        perception.warping(
            mug_pcd, tree_pcd, canon_mug, canon_tree, source_any_rotation=args.any_rotation
        )
    elif args.task == "bowl_on_mug":
        bowl_pcd, mug_pcd = perception.bowl_mug_segmentation(cloud)

        canon_bowl = utils.CanonObj.from_pickle(constants.NDF_BOWLS_PCA_PATH)
        canon_mug = utils.CanonObj.from_pickle(constants.NDF_MUGS_PCA_PATH)
        canon_bowl.init_scale = constants.NDF_BOWLS_INIT_SCALE
        canon_mug.init_scale = constants.NDF_MUGS_INIT_SCALE

        perception.warping(
            bowl_pcd, mug_pcd, canon_bowl, canon_mug, source_any_rotation=args.any_rotation
        )
    elif args.task == "bottle_in_box":
        bottle_pcd, box_pcd = perception.bottle_box_segmentation(cloud)

        canon_bottle = utils.CanonObj.from_pickle(constants.NDF_BOTTLES_PCA_PATH)
        canon_box = utils.CanonObj.from_pickle(constants.BOXES_PCA_PATH)
        canon_bottle.init_scale = constants.NDF_BOTTLES_INIT_SCALE
        canon_box.init_scale = constants.BOXES_INIT_SCALE

        perception.warping(
            bottle_pcd, box_pcd, canon_bottle, canon_box, source_any_rotation=args.any_rotation
        )
    else:
        raise ValueError("Invalid task.")


parser = argparse.ArgumentParser("Find objects for a particular task, create warps.")
parser.add_argument("task", help="[mug_tree, bowl_on_mug, bottle_in_box]")
parser.add_argument("-a", "--any-rotation", default=False, action="store_true")
main(parser.parse_args())
