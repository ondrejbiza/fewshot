import argparse
import subprocess
import threading
from typing import Tuple, Dict, Any
import time
import rospy
import open3d as o3d
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import trimesh
import pybullet as pb
from scipy.spatial.transform import Rotation

from pybullet_planning.pybullet_tools import utils as pu
from online_isec.point_cloud_proxy import PointCloudProxy, RealsenseStructurePointCloudProxy
import online_isec.utils as isec_utils
from online_isec.ur5 import UR5
from online_isec import constants
import utils
import viz_utils


def main():

    rospy.init_node("easy_perception")
    pc_proxy = RealsenseStructurePointCloudProxy()
    ur5 = UR5(setup_planning=True)
    time.sleep(2)
    ur5.move_to_j(ur5.home_joint_values)

    cloud = pc_proxy.get_all()
    assert cloud is not None
    cloud = isec_utils.mask_workspace(cloud, (*pc_proxy.desk_center, pc_proxy.z_min + 0.02))
    mug_pc, tree_pc = isec_utils.find_mug_and_tree(cloud)
    max_size = 2000
    if len(mug_pc) > max_size:
        mug_pc, _ = utils.farthest_point_sample(mug_pc, max_size)
    if len(tree_pc) > max_size:
        tree_pc, _ = utils.farthest_point_sample(tree_pc, max_size)

    # load canonical objects
    with open("data/ndf_mugs_pca_4_dim.npy", "rb") as f:
        canon_mug = pickle.load(f)
    with open("data/real_tree_pc.pkl", "rb") as f:
        canon_tree = pickle.load(f)

    mug_pc_complete, _, mug_param = utils.planar_pose_warp_gd(canon_mug["pca"], canon_mug["canonical_obj"], mug_pc, object_size_reg=0.1, n_angles=12)
    tree_pc_complete, _, tree_param = utils.planar_pose_gd(canon_tree["canonical_obj"], tree_pc, n_angles=12)

    with open("data/real_pick_clone.pkl", "rb") as f:
        data = pickle.load(f)
        index, target_quat = data["index"], data["quat"]
        target_pos = mug_pc_complete[index]

    target_pos = target_pos + constants.DESK_CENTER

    print("base to tool0_controller")
    print("Target pos: {}".format(target_pos))
    print("Target quat: {}".format(target_quat))

    T_b_to_g_prime = utils.pos_quat_to_transform(target_pos, target_quat)
    T_bl_to_b = np.linalg.inv(ur5.tf_proxy.lookup_transform("base_link", "base"))
    T_g_to_b = np.linalg.inv(ur5.tf_proxy.lookup_transform("tool0_controller", "base"))
    T_b_to_f = ur5.tf_proxy.lookup_transform("flange", "base")

    T = T_bl_to_b @ T_b_to_g_prime @ T_g_to_b @ T_b_to_f
    target_pos, target_quat = utils.transform_to_pos_quat(T)

    print("base_link to flange")
    print("Target pos: {}".format(target_pos))
    print("Target quat: {}".format(target_quat))

    input("big red button")
    ur5.plan_and_execute_pose_target(target_pos, target_quat)


main()
