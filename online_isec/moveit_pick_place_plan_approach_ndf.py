import argparse
import os
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import torch
import pickle
import rospy

import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.utils import path_util, torch_util, trimesh_util
from ndf_robot.utils import util, trimesh_util
from ndf_robot.opt.optimizer import OccNetOptimizer

import online_isec.utils as isec_utils
from online_isec.point_cloud_proxy_sync import RealsenseStructurePointCloudProxy
from online_isec import perception
from online_isec.show_ndf_demonstration import show_pose
from online_isec import constants
from online_isec.ur5 import UR5
import utils


def pick(
    model, new_mug_pc: NDArray, pick_load_path: str, num_samples: int, sigma: float, opt_iterations: int,
    show: bool=False) -> Tuple[NDArray, NDArray]:

    T_ws_to_b = isec_utils.workspace_to_base()

    with open(pick_load_path, "rb") as f:
        x = pickle.load(f)
        mug_pc = x["observed_pc"]
        T_g_pick_to_b = x["T_g_to_b"]
        T_g_pick_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), T_g_pick_to_b)

    # Move pick location up to be closer to the rim.
    T_g_pick_up_to_ws = utils.pos_quat_to_transform(*utils.move_hand_back(utils.transform_to_pos_quat(T_g_pick_to_ws), -0.02))

    pick_reference_points = np.random.normal(T_g_pick_up_to_ws[:3, 3], sigma, size=(num_samples, 3))

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(mug_pc[:, 0], mug_pc[:, 1], mug_pc[:, 2], alpha=0.5, c="blue")
        ax.scatter(pick_reference_points[:, 0], pick_reference_points[:, 1], pick_reference_points[:, 2], c="gray")
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0.0, 0.4)
        plt.show()

    grasp_optimizer = OccNetOptimizer(
        model,
        query_pts=pick_reference_points,  # TODO: what is this?
        # query_pts_real_shape=optimizer_gripper_pts_rs,  # TODO: what is this?
        opt_iterations=opt_iterations)
    grasp_optimizer.set_demo_info([{
        "demo_obj_pts": mug_pc,
        "demo_query_pts": pick_reference_points,
    }])

    transforms_list, best_idx = grasp_optimizer.optimize_transform_implicit(new_mug_pc, ee=True)
    tmp = util.pose_stamped2list(util.pose_from_matrix(transforms_list[best_idx]))
    T_rel = utils.pos_quat_to_transform(tmp[:3], tmp[3:])

    T_g_pick_new_to_ws = T_rel @ T_g_pick_to_ws

    # Move pick location back down, makes it more reliable..
    T_g_pick_new_pre_to_ws = utils.pos_quat_to_transform(*utils.move_hand_back(utils.transform_to_pos_quat(T_g_pick_new_to_ws), -0.05))

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(new_mug_pc[:, 0], new_mug_pc[:, 1], new_mug_pc[:, 2], alpha=0.5, c="blue")
        show_pose(ax, T_g_pick_new_to_ws)
        show_pose(ax, T_g_pick_new_pre_to_ws)
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0.0, 0.4)
        plt.show()

    return T_g_pick_new_to_ws, T_g_pick_new_pre_to_ws


def place(
    T_g_pick_new_to_ws: NDArray, model, new_mug_pc: NDArray, pick_load_path: str, place_load_path: str, num_samples: int,
    sigma: float, opt_iterations: int, show: bool=False) -> Tuple[NDArray, NDArray]:

    T_ws_to_b = isec_utils.workspace_to_base()

    with open(pick_load_path, "rb") as f:
        x = pickle.load(f)
        mug_pc = x["observed_pc"]
        T_g_pick_to_b = x["T_g_to_b"]
        T_g_pick_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), T_g_pick_to_b)

    with open(place_load_path, "rb") as f:
        x = pickle.load(f)
        tree_pc = x["observed_pc"]
        T_g_place_to_b = x["T_g_to_b"] 
        T_g_place_pre_to_g = x["T_g_pre_to_g"] 
        T_g_place_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), T_g_place_to_b)

    T_rel = T_g_place_to_ws @ np.linalg.inv(T_g_pick_to_ws)
    mug_pc_on_tree = utils.transform_pointcloud_2(mug_pc, T_rel)

    place_reference_points = np.random.normal(constants.NDF_BRANCH_POSITION, sigma, size=(num_samples, 3))

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(mug_pc_on_tree[:, 0], mug_pc_on_tree[:, 1], mug_pc_on_tree[:, 2], alpha=0.5, c="blue")
        ax.scatter(tree_pc[:, 0], tree_pc[:, 1], tree_pc[:, 2], alpha=0.5, c="brown")
        ax.scatter(place_reference_points[:, 0], place_reference_points[:, 1], place_reference_points[:, 2], c="gray")
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0.0, 0.4)
        plt.show()

    place_optimizer = OccNetOptimizer(
        model,
        query_pts=place_reference_points,  # TODO: what is this?
        # query_pts_real_shape=optimizer_gripper_pts_rs,  # TODO: what is this?
        opt_iterations=opt_iterations)
    place_optimizer.set_demo_info([{
        "demo_obj_pts": mug_pc_on_tree,
        "demo_query_pts": place_reference_points,
    }])

    tmp, best_idx = place_optimizer.optimize_transform_implicit(new_mug_pc, ee=False)
    tmp = util.pose_stamped2list(util.pose_from_matrix(tmp[best_idx]))
    T_rel = utils.pos_quat_to_transform(tmp[:3], tmp[3:])

    T_g_place_new_to_ws = T_rel @ T_g_pick_new_to_ws
    T_g_place_new_pre_to_ws = T_g_place_new_to_ws @ T_g_place_pre_to_g

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        tmp = utils.transform_pointcloud_2(mug_pc, T_rel)
        ax.scatter(tmp[:, 0], tmp[:, 1], tmp[:, 2], alpha=0.5, c="blue")
        ax.scatter(tree_pc[:, 0], tree_pc[:, 1], tree_pc[:, 2], alpha=0.5, c="brown")
        show_pose(ax, T_g_place_new_to_ws)
        show_pose(ax, T_g_place_new_pre_to_ws)
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0.0, 0.4)
        plt.show()

    return T_g_place_new_to_ws, T_g_place_new_pre_to_ws


def main(args):

    rospy.init_node("moveit_pick_place_plan_approach_ndf")

    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)
    ur5.gripper.open_gripper(position=70)

    pc_proxy = RealsenseStructurePointCloudProxy()

    num_points = 1500  # Maximum NDFs can handle.
    new_mug_pc, new_tree_pc = perception.mug_tree_simple_perception(pc_proxy, num_points)

    model_path = os.path.join(path_util.get_ndf_model_weights(), "multi_category_weights.pth")  
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type="pointnet", return_features=True, sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))

    num_samples = 500
    sigma = 0.02
    opt_iterations = 500

    T_ws_to_b = isec_utils.workspace_to_base()

    target_pick_T_g_to_ws, target_pick_T_g_to_ws_pre = pick(model, new_mug_pc, args.pick_load_path, num_samples, sigma, opt_iterations)

    target_pick_T_g_to_b = np.matmul(T_ws_to_b, target_pick_T_g_to_ws)
    target_pick_T_g_to_b_pre = np.matmul(T_ws_to_b, target_pick_T_g_to_ws_pre)

    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(target_pick_T_g_to_b_pre))
    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(target_pick_T_g_to_b))
    ur5.gripper.close_gripper()
    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(target_pick_T_g_to_b_pre))
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    target_place_T_g_to_ws, target_place_T_g_to_ws_pre = place(target_pick_T_g_to_ws, model, new_mug_pc, args.pick_load_path, args.place_load_path, num_samples, sigma, opt_iterations)

    target_place_T_g_to_b = np.matmul(T_ws_to_b, target_place_T_g_to_ws)
    target_place_T_g_to_b_pre = np.matmul(T_ws_to_b, target_place_T_g_to_ws_pre)

    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(target_place_T_g_to_b_pre))
    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(target_place_T_g_to_b))
    ur5.gripper.open_gripper()
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick-load-path", default="data/230201_real_pick_clone_full.pkl")
    parser.add_argument("--place-load-path", default="data/230201_real_place_clone_full.pkl")
    main(parser.parse_args())
