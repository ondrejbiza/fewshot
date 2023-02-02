import argparse
import os
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
import utils


def pick(model, new_mug_pc: NDArray, pick_load_path: str, num_samples: int, sigma: float, opt_iterations: int):

    T_ws_to_b = isec_utils.workspace_to_base()

    with open(pick_load_path, "rb") as f:
        x = pickle.load(f)
        mug_pc = x["observed_pc"]
        pick_T_g_to_b = x["T_g_to_b"]
        pick_T_g_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), pick_T_g_to_b)

    pick_reference_points = np.random.normal(pick_T_g_to_ws[:3, 3], sigma, size=(num_samples, 3))

    grasp_optimizer = OccNetOptimizer(
        model,
        query_pts=pick_reference_points,  # TODO: what is this?
        # query_pts_real_shape=optimizer_gripper_pts_rs,  # TODO: what is this?
        opt_iterations=opt_iterations)
    grasp_optimizer.set_demo_info([{
        "demo_obj_pts": mug_pc,
        "demo_query_pts": pick_reference_points,
    }])

    tmp, best_idx = grasp_optimizer.optimize_transform_implicit(new_mug_pc, ee=True)
    tmp = util.pose_stamped2list(util.pose_from_matrix(tmp[best_idx]))
    pos, quat = tmp[:3], tmp[3:]
    new_T = utils.pos_quat_to_transform(pos, quat)
    target_pick_T_g_to_ws = np.matmul(new_T, pick_T_g_to_ws)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(new_mug_pc[:, 0], new_mug_pc[:, 1], new_mug_pc[:, 2], alpha=0.5, c="blue")
    show_pose(ax, target_pick_T_g_to_ws)
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(0.0, 0.4)
    plt.show()


def place(model, new_mug_pc: NDArray, pick_load_path: str, place_load_path: str, num_samples: int, sigma: float, opt_iterations: int):

    T_ws_to_b = isec_utils.workspace_to_base()

    with open(pick_load_path, "rb") as f:
        x = pickle.load(f)
        mug_pc = x["observed_pc"]
        pick_T_g_to_b = x["T_g_to_b"]
        pick_T_g_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), pick_T_g_to_b)

    with open(place_load_path, "rb") as f:
        x = pickle.load(f)
        tree_pc = x["observed_pc"]
        place_T_g_to_b = x["T_g_to_b"] 
        place_T_g_pre_to_g = x["T_g_pre_to_g"] 
        place_T_g_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), place_T_g_to_b)

    place_T_to_pick_T = np.matmul(pick_T_g_to_ws, np.linalg.inv(place_T_g_to_ws))
    pos = np.array(constants.NDF_BRANCH_POSITION)[None]
    pos = utils.transform_pointcloud_2(pos, place_T_to_pick_T)[0]
    place_reference_points = np.random.normal(pos, sigma, size=(num_samples, 3))

    place_optimizer = OccNetOptimizer(
        model,
        query_pts=place_reference_points,  # TODO: what is this?
        # query_pts_real_shape=optimizer_gripper_pts_rs,  # TODO: what is this?
        opt_iterations=opt_iterations)
    place_optimizer.set_demo_info([{
        "demo_obj_pts": mug_pc,
        "demo_query_pts": place_reference_points,
    }])

    tmp, best_idx = place_optimizer.optimize_transform_implicit(new_mug_pc, ee=True)
    tmp = util.pose_stamped2list(util.pose_from_matrix(tmp[best_idx]))
    pos, quat = tmp[:3], tmp[3:]
    new_T = utils.pos_quat_to_transform(pos, quat)
    target_place_T_g_to_ws = np.matmul(new_T, place_T_g_to_ws)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(tree_pc[:, 0], tree_pc[:, 1], tree_pc[:, 2], alpha=0.5, c="blue")
    ax.scatter(place_reference_points[:, 0], place_reference_points[:, 1], place_reference_points[:, 2], c="gray")
    show_pose(ax, target_place_T_g_to_ws)
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(0.0, 0.4)
    plt.show()


def main(args):

    rospy.init_node("moveit_plan_pick_place_plan_approach")
    pc_proxy = RealsenseStructurePointCloudProxy()

    num_points = 1500  # Maximum NDFs can handle.
    new_mug_pc, new_tree_pc = perception.mug_tree_simple_perception(pc_proxy, num_points)

    model_path = os.path.join(path_util.get_ndf_model_weights(), "multi_category_weights.pth")  
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type="pointnet", return_features=True, sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))

    num_samples = 500
    sigma = 0.02
    opt_iterations = 500

    pick(model, new_mug_pc, args.pick_load_path, num_samples, sigma, opt_iterations)
    place(model, new_mug_pc, args.pick_load_path, args.place_load_path, num_samples, sigma, opt_iterations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick-load-path", default="data/230201_real_pick_clone_full.pkl")
    parser.add_argument("--place-load-path", default="data/230201_real_place_clone_full.pkl")
    main(parser.parse_args())
