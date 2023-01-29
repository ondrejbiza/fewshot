import argparse
import numpy as np
import rospy

from online_isec import constants
from online_isec import perception
from online_isec.moveit_plan_pick_place_plan import pick
from online_isec.point_cloud_proxy import RealsenseStructurePointCloudProxy
from online_isec.ur5 import UR5


def main(args):

    rospy.init_node("moveit_plan_pick")
    pc_proxy = RealsenseStructurePointCloudProxy()

    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    cloud = pc_proxy.get_all()
    assert cloud is not None

    if args.pca_8_dim:
        canon_mug_path = "data/ndf_mugs_pca_8_dim.npy"
    else:
        canon_mug_path = "data/ndf_mugs_pca_4_dim.npy"

    mug_pc_complete, mug_param, tree_pc_complete, tree_param, canon_mug, canon_tree = perception.mug_tree_perception(
        pc_proxy, np.array(constants.DESK_CENTER), ur5.tf_proxy, ur5.moveit_scene,
        add_mug_to_planning_scene=False, add_tree_to_planning_scene=False, rviz_pub=ur5.rviz_pub,
        canon_mug_path=canon_mug_path, ablate_no_mug_warping=args.ablate_no_mug_warping,
        close_proxy=True
    )

    pick(mug_pc_complete, mug_param, ur5, safe_release=True)

    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


parser = argparse.ArgumentParser()
parser.add_argument("--pca-8-dim", default=False, action="store_true")
parser.add_argument("--ablate-no-mug-warping", default=False, action="store_true")
main(parser.parse_args())
