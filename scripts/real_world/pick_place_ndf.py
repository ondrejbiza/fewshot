import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import rospy
import torch

import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.opt.optimizer import OccNetOptimizer
from ndf_robot.utils import path_util, torch_util, trimesh_util
from ndf_robot.utils import util, trimesh_util

from src import utils, viz_utils
from src.real_world import constants, perception
import src.real_world.utils as rw_utils
from src.real_world.point_cloud_proxy import PointCloudProxy
from src.real_world.ur5 import UR5

EASY_MOTION_TRIES = 1
HARD_MOTION_TRIES = 5


def draw_arrow(ax, orig, delta, color):

    ax.quiver(
        orig[0], orig[1], orig[2], # <-- starting point of vector
        delta[0], delta[1], delta[2], # <-- directions of vector
        color=color, alpha=0.8, lw=3,
    )


def show_pose(ax, T):

    orig = T[:3, 3]
    rot = T[:3, :3]
    x_arrow = np.matmul(rot, np.array([0.05, 0., 0.]))
    y_arrow = np.matmul(rot, np.array([0., 0.05, 0.]))
    z_arrow = np.matmul(rot, np.array([0., 0., 0.05]))
    draw_arrow(ax, orig, x_arrow, "red")
    draw_arrow(ax, orig, y_arrow, "green")
    draw_arrow(ax, orig, z_arrow, "blue")


def pick(
    model, observed_source_pcd: NDArray, pick_load_paths: List[str], num_samples: int,
    sigma: float, opt_iterations: int, show: bool=False) -> Tuple[NDArray, NDArray]:

    trans_ws_to_b = rw_utils.workspace_to_base()

    assert len(pick_load_paths) == 1, "Multiple demonstrations not yet implemented."
    with open(pick_load_paths[0], "rb") as f:
        x = pickle.load(f)
        demo_source_pcd = x["observed_pc"]
        trans_t0_to_b = x["trans_t0_to_b"]
        trans_pre_t0_to_t0 = x["trans_pre_t0_to_t0"]

        trans_t0_to_ws = np.matmul(np.linalg.inv(trans_ws_to_b), trans_t0_to_b)

    pos, _ = utils.transform_to_pos_quat(trans_t0_to_ws)
    pick_reference_points = np.random.normal(pos, sigma, size=(num_samples, 3))

    if show:
        viz_utils.show_pcds_plotly({
            "demo_source_pcd": demo_source_pcd,
            "pick_reference_points": pick_reference_points
        })

    grasp_optimizer = OccNetOptimizer(
        model,
        query_pts=pick_reference_points,  # TODO: what is this?
        # query_pts_real_shape=optimizer_gripper_pts_rs,  # TODO: what is this?
        opt_iterations=opt_iterations)
    grasp_optimizer.set_demo_info([{
        "demo_obj_pts": demo_source_pcd,
        "demo_query_pts": pick_reference_points,
    }])

    transforms_list, best_idx = grasp_optimizer.optimize_transform_implicit(observed_source_pcd, ee=True)
    tmp = util.pose_stamped2list(util.pose_from_matrix(transforms_list[best_idx]))
    trans_rel = utils.pos_quat_to_transform(tmp[:3], tmp[3:])

    trans_pick_t0_to_ws = trans_rel @ trans_t0_to_ws
    trans_pre_pick_t0_to_ws = trans_pick_t0_to_ws @ trans_pre_t0_to_t0

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(observed_source_pcd[:, 0], observed_source_pcd[:, 1], observed_source_pcd[:, 2], alpha=0.5, c="blue")
        show_pose(ax, trans_pick_t0_to_ws)
        show_pose(ax, trans_pre_pick_t0_to_ws)
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0.0, 0.4)
        plt.show()

    return trans_pick_t0_to_ws, trans_pre_pick_t0_to_ws


def place(
    trans_new_pick_t0_to_ws: NDArray, model, observed_source_pcd: NDArray, pick_load_paths: List[str],
    place_load_paths: List[str], num_samples: int, sigma: float, opt_iterations: int,
    show: bool=False) -> Tuple[NDArray, NDArray]:

    trans_ws_to_b = rw_utils.workspace_to_base()

    assert len(pick_load_paths) == 1, "Multiple demonstrations not yet implemented."
    with open(pick_load_paths[0], "rb") as f:
        x = pickle.load(f)
        demo_source_pcd = x["observed_pc"]
        trans_pick_t0_to_b = x["trans_t0_to_b"]

    trans_pick_t0_to_ws = np.matmul(np.linalg.inv(trans_ws_to_b), trans_pick_t0_to_b)

    assert len(pick_load_paths) == 1, "Multiple demonstrations not yet implemented."
    with open(place_load_paths[0], "rb") as f:
        x = pickle.load(f)
        demo_target_pcd = x["observed_pc"]
        trans_place_t0_to_b = x["trans_t0_to_b"] 
        trans_place_pre_t0_to_b = x["trans_pre_t0_to_b"] 

    trans_place_pre_t0_to_t0 = np.linalg.inv(trans_place_t0_to_b) @ trans_place_pre_t0_to_b
    trans_place_t0_to_ws = np.matmul(np.linalg.inv(trans_ws_to_b), trans_place_t0_to_b)

    trans_rel = trans_place_t0_to_ws @ np.linalg.inv(trans_pick_t0_to_ws)
    demo_source_pcd_on_target = utils.transform_pcd(demo_source_pcd, trans_rel)

    place_reference_points = np.random.normal(constants.NDF_BRANCH_POSITION, sigma, size=(num_samples, 3))

    if show:
        viz_utils.show_pcds_plotly({
            "source_pcd": demo_source_pcd_on_target,
            "target_pcd": demo_target_pcd,
            "place_reference_points": place_reference_points,
        })

    place_optimizer = OccNetOptimizer(
        model,
        query_pts=place_reference_points,  # TODO: what is this?
        # query_pts_real_shape=optimizer_gripper_pts_rs,  # TODO: what is this?
        opt_iterations=opt_iterations)
    place_optimizer.set_demo_info([{
        "demo_obj_pts": demo_source_pcd_on_target,
        "demo_query_pts": place_reference_points,
    }])

    tmp, best_idx = place_optimizer.optimize_transform_implicit(observed_source_pcd, ee=False)
    tmp = util.pose_stamped2list(util.pose_from_matrix(tmp[best_idx]))
    trans_rel = utils.pos_quat_to_transform(tmp[:3], tmp[3:])

    trans_new_place_t0_to_ws = trans_rel @ trans_new_pick_t0_to_ws
    trans_new_pre_place_t0_to_ws = trans_new_place_t0_to_ws @ trans_place_pre_t0_to_t0

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        tmp = utils.transform_pcd(observed_source_pcd, trans_rel)
        ax.scatter(tmp[:, 0], tmp[:, 1], tmp[:, 2], alpha=0.5, c="blue")
        ax.scatter(demo_target_pcd[:, 0], demo_target_pcd[:, 1], demo_target_pcd[:, 2], alpha=0.5, c="brown")
        show_pose(ax, trans_new_place_t0_to_ws)
        show_pose(ax, trans_new_pre_place_t0_to_ws)
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0.0, 0.4)
        plt.show()

    return trans_new_place_t0_to_ws, trans_new_pre_place_t0_to_ws


def main(args):

    rospy.init_node("pick_place_ndf")
    pc_proxy = PointCloudProxy()

    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)
    ur5.gripper.open_gripper(position=20)

    platform_pcd = None
    if args.platform:
        cloud = pc_proxy.get_all()
        assert cloud is not None
        platform_pcd = perception.platform_segmentation(cloud)
        input("Platform captured. Continue? ")

    cloud = pc_proxy.get_all()

    num_points = 1500  # Maximum NDFs can handle.

    if args.task == "mug_tree":
        source_pcd, target_pcd = perception.mug_tree_segmentation(pc_proxy, num_points, platform_pcd=platform_pcd)
    elif args.task == "bowl_on_mug":
        source_pcd, target_pcd = perception.bowl_mug_segmentation(pc_proxy, num_points, platform_pcd=platform_pcd)
    elif args.task == "bottle_in_box":
        source_pcd, target_pcd = perception.bottle_box_segmentation(pc_proxy, num_points, platform_pcd=platform_pcd)
    else:
        raise ValueError("Unknown task.")

    model_path = os.path.join(path_util.get_ndf_model_weights(), "multi_category_weights.pth")  
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type="pointnet", return_features=True, sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))

    num_samples = 500
    sigma = 0.02
    opt_iterations = 500

    trans_ws_to_b = rw_utils.workspace_to_base()

    trans_pick_t0_to_ws, trans_pre_pick_t0_to_ws = pick(model, source_pcd, args.pick_load_paths, num_samples, sigma, opt_iterations)

    trans_pick_t0_to_b = np.matmul(trans_ws_to_b, trans_pick_t0_to_ws)
    trans_pre_pick_t0_to_b = np.matmul(trans_ws_to_b, trans_pre_pick_t0_to_ws)

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_pick_t0_to_b), num_plans=HARD_MOTION_TRIES)
    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pick_t0_to_b), num_plans=EASY_MOTION_TRIES)
    ur5.gripper.close_gripper()
    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_pick_t0_to_b), num_plans=HARD_MOTION_TRIES)  # TODO: Post-pick pose and in-hand image.
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    trans_place_t0_to_ws, trans_pre_place_t0_to_ws = place(trans_pick_t0_to_ws, model, source_pcd, args.pick_load_path, args.place_load_path, num_samples, sigma, opt_iterations)

    trans_place_t0_to_b = np.matmul(trans_ws_to_b, trans_place_t0_to_ws)
    trans_pre_place_t0_to_b = np.matmul(trans_ws_to_b, trans_pre_place_t0_to_ws)

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_place_t0_to_b), num_plans=HARD_MOTION_TRIES)
    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_place_t0_to_b), num_plans=EASY_MOTION_TRIES)
    ur5.gripper.open_gripper()
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="[mug_tree, bowl_on_mug, bottle_in_box]")
    parser.add_argument("pick_load_paths", type=str, nargs="+")
    parser.add_argument("place_load_paths", type=str, nargs="+")

    parser.add_argument("-p", "--platform", default=False, action="store_true",
                        help="First take a point cloud of a platform. Then subtract the platform from the next point cloud.")

    main(parser.parse_args())
