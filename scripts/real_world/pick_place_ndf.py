import argparse
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import rospy
import torch

import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.opt.optimizer import OccNetOptimizer
from ndf_robot.utils import path_util, util

from src import utils, viz_utils
from src.real_world import constants, perception
import src.real_world.utils as rw_utils
from src.real_world.point_cloud_proxy import PointCloudProxy
from src.real_world.ur5 import UR5
from src.real_world.simulation import Simulation

EASY_MOTION_TRIES = 1
HARD_MOTION_TRIES = 5


def ndf_prepare_pick_demos(pick_load_paths: List[str], num_samples: int, sigma: float, show: bool=False
                           ) -> Tuple[NDArray[np.float64], NDArray[np.float64],
                                      NDArray[np.float32], List[Dict[str, Any]]]:

    assert len(pick_load_paths) > 0

    first_trans_pre_t0_to_t0 = None
    first_pick_reference_points = None
    first_trans_t0_tip_to_ws = None

    trans_ws_to_b = rw_utils.workspace_to_base()

    # Process all demonstrations.
    demos_list = []

    for i in range(len(pick_load_paths)):
        with open(pick_load_paths[i], "rb") as f:
            x = pickle.load(f)
            demo_source_pcd = x["observed_pc"]
            trans_t0_to_b = x["trans_t0_to_b"]
            trans_pre_t0_to_t0 = x["trans_pre_t0_to_t0"]

        trans_t0_to_ws = np.matmul(np.linalg.inv(trans_ws_to_b), trans_t0_to_b)
        trans_t0_tip_to_ws = trans_t0_to_ws @ rw_utils.tool0_tip_to_tool0()

        pos, _ = utils.transform_to_pos_quat(trans_t0_tip_to_ws)
        pick_reference_points = np.random.normal(pos, sigma, size=(num_samples, 3)).astype(np.float32)

        demos_list.append({
            "demo_obj_pts": demo_source_pcd,
            "demo_query_pts": pick_reference_points,
        })

        if i == 0:
            first_trans_pre_t0_to_t0 = trans_pre_t0_to_t0
            first_pick_reference_points = pick_reference_points
            first_trans_t0_tip_to_ws = trans_t0_tip_to_ws

        if show:
            viz_utils.show_pcds_plotly({
                "demo_source_pcd": demo_source_pcd,
                "pick_reference_points": pick_reference_points
            })

    return first_trans_t0_tip_to_ws, first_trans_pre_t0_to_t0, first_pick_reference_points, demos_list


def ndf_prepare_place_demos(pick_load_paths: List[str], place_load_paths: List[str],
                            num_samples: int, sigma: float, show: bool=False
                            ) -> Tuple[NDArray[np.float64], NDArray[np.float32],
                                       NDArray[np.float32], List[Dict[str, Any]]]:

    assert len(pick_load_paths) > 0
    assert len(pick_load_paths) == len(place_load_paths)

    first_place_reference_points = None
    first_trans_place_pre_t0_to_t0 = None
    first_demo_target_pcd = None

    demos_list = []

    for i in range(len(pick_load_paths)):

        trans_ws_to_b = rw_utils.workspace_to_base()

        with open(pick_load_paths[i], "rb") as f:
            x = pickle.load(f)
            demo_source_pcd = x["observed_pc"]
            trans_pick_t0_to_b = x["trans_t0_to_b"]

        trans_pick_t0_to_ws = np.matmul(np.linalg.inv(trans_ws_to_b), trans_pick_t0_to_b)

        with open(place_load_paths[i], "rb") as f:
            x = pickle.load(f)
            demo_target_pcd = x["observed_pc"]
            trans_place_t0_to_b = x["trans_t0_to_b"] 
            trans_place_pre_t0_to_b = x["trans_pre_t0_to_b"] 

        trans_place_pre_t0_to_t0 = np.linalg.inv(trans_place_t0_to_b) @ trans_place_pre_t0_to_b
        trans_place_t0_to_ws = np.matmul(np.linalg.inv(trans_ws_to_b), trans_place_t0_to_b)

        trans_rel = trans_place_t0_to_ws @ np.linalg.inv(trans_pick_t0_to_ws)
        demo_source_pcd_on_target = utils.transform_pcd(demo_source_pcd, trans_rel)

        place_reference_points = np.random.normal(constants.NDF_BRANCH_POSITION, sigma, size=(num_samples, 3))

        demos_list.append({
            "demo_obj_pts": demo_source_pcd_on_target,
            "demo_query_pts": place_reference_points,
        })

        if i == 0:
            first_place_reference_points = place_reference_points
            first_trans_place_pre_t0_to_t0 = trans_place_pre_t0_to_t0
            first_demo_target_pcd = demo_target_pcd

        if show:
            viz_utils.show_pcds_plotly({
                "source_pcd": demo_source_pcd_on_target,
                "target_pcd": demo_target_pcd,
                "place_reference_points": place_reference_points,
            })

    return first_trans_place_pre_t0_to_t0, first_place_reference_points, first_demo_target_pcd, demos_list


def pick(model, observed_source_pcd: NDArray, pick_load_paths: List[str],
         num_samples: int, sigma: float, opt_iterations: int, show: bool=False
         ) -> Tuple[NDArray, NDArray]:

    trans_t0_tip_to_ws, trans_pre_t0_to_t0, pick_reference_points, demos_list = ndf_prepare_pick_demos(
        pick_load_paths, num_samples, sigma, show=show)

    grasp_optimizer = OccNetOptimizer(
        model,
        query_pts=pick_reference_points,
        opt_iterations=opt_iterations
    )
    grasp_optimizer.set_demo_info(demos_list)

    transforms_list, best_idx = grasp_optimizer.optimize_transform_implicit(observed_source_pcd, ee=True)
    tmp = util.pose_stamped2list(util.pose_from_matrix(transforms_list[best_idx]))
    trans_rel = utils.pos_quat_to_transform(tmp[:3], tmp[3:])

    trans_pick_t0_tip_to_ws = trans_rel @ trans_t0_tip_to_ws
    trans_pick_t0_to_ws = trans_pick_t0_tip_to_ws @ np.linalg.inv(rw_utils.tool0_tip_to_tool0())

    trans_pre_pick_t0_to_ws = trans_pick_t0_to_ws @ trans_pre_t0_to_t0

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(observed_source_pcd[:, 0], observed_source_pcd[:, 1], observed_source_pcd[:, 2], alpha=0.5, c="blue")
        viz_utils.show_pose(ax, trans_pick_t0_to_ws)
        viz_utils.show_pose(ax, trans_pre_pick_t0_to_ws)
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0.0, 0.4)
        plt.show()

    return trans_pick_t0_to_ws, trans_pre_pick_t0_to_ws


def place(trans_new_pick_t0_to_ws: NDArray, model, observed_source_pcd: NDArray,
          pick_load_paths: List[str], place_load_paths: List[str], num_samples: int,
          sigma: float, opt_iterations: int, show: bool=False
          ) -> Tuple[NDArray, NDArray]:

    trans_place_pre_t0_to_t0, place_reference_points, demo_target_pcd, demos_list = ndf_prepare_place_demos(
        pick_load_paths, place_load_paths, num_samples, sigma, show)

    place_optimizer = OccNetOptimizer(
        model,
        query_pts=place_reference_points,  # TODO: what is this?
        # query_pts_real_shape=optimizer_gripper_pts_rs,  # TODO: what is this?
        opt_iterations=opt_iterations)
    place_optimizer.set_demo_info(demos_list)

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
        viz_utils.show_pose(ax, trans_new_place_t0_to_ws)
        viz_utils.show_pose(ax, trans_new_pre_place_t0_to_ws)
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

    sim = Simulation()

    num_points = 1500  # Maximum NDFs can handle.

    if args.task == "mug_tree":
        source_pcd, target_pcd = perception.mug_tree_segmentation(cloud, num_points, platform_pcd=platform_pcd)
    elif args.task == "bowl_on_mug":
        source_pcd, target_pcd = perception.bowl_mug_segmentation(cloud, num_points, platform_pcd=platform_pcd)
    elif args.task == "bottle_in_box":
        source_pcd, target_pcd = perception.bottle_box_segmentation(cloud, num_points, platform_pcd=platform_pcd)
    else:
        raise ValueError("Unknown task.")

    model_path = os.path.join(path_util.get_ndf_model_weights(), "multi_category_weights.pth")  
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type="pointnet", return_features=True, sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))

    num_samples = 500
    sigma = 0.02
    opt_iterations = 500

    trans_ws_to_b = rw_utils.workspace_to_base()

    trans_pick_t0_to_ws, trans_pre_pick_t0_to_ws = pick(
        model, source_pcd, args.pick_load_paths, num_samples,
        sigma, opt_iterations, show=False)

    trans_post_t0_to_ws = rw_utils.get_post_pick_pose(trans_pick_t0_to_ws)
    trans_post_t0_to_b = rw_utils.workspace_to_base() @ trans_post_t0_to_ws

    trans_pick_t0_to_b = np.matmul(trans_ws_to_b, trans_pick_t0_to_ws)
    trans_pre_pick_t0_to_b = np.matmul(trans_ws_to_b, trans_pre_pick_t0_to_ws)

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_pick_t0_to_b), num_plans=HARD_MOTION_TRIES)
    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pick_t0_to_b), num_plans=EASY_MOTION_TRIES)
    ur5.gripper.close_gripper()
    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_post_t0_to_b), num_plans=HARD_MOTION_TRIES)

    if args.no_in_hand:
        in_hand = source_pcd
        trans_t0_to_ws = trans_pick_t0_to_ws
    else:
        # Get a new point cloud with the mug in hand. Remove all points that belong to the hand.
        cloud = pc_proxy.get_all()
        robotiq_id = sim.add_object("data/robotiq.urdf", np.array([0., 0., 0.]), np.array([0., 0., 0., 1.]))
        in_hand = perception.in_hand_segmentation(cloud)
        in_hand = perception.clean_up_in_hand_image(in_hand, ur5, robotiq_id, sim, num_points=2000)
        sim.remove_object(robotiq_id)

        # Get a new pose of the hand that corresponds to the point cloud we took above.
        trans_t0_to_b = utils.pos_quat_to_transform(*ur5.get_tool0_to_base())
        trans_t0_to_ws = np.linalg.inv(rw_utils.workspace_to_base()) @ trans_t0_to_b

    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    trans_place_t0_to_ws, trans_pre_place_t0_to_ws = place(
        trans_t0_to_ws, model, in_hand, args.pick_load_paths, args.place_load_paths,
        num_samples, sigma, opt_iterations, show=False)

    trans_place_t0_to_b = np.matmul(trans_ws_to_b, trans_place_t0_to_ws)
    trans_pre_place_t0_to_b = np.matmul(trans_ws_to_b, trans_pre_place_t0_to_ws)

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_place_t0_to_b), num_plans=HARD_MOTION_TRIES)
    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_place_t0_to_b), num_plans=EASY_MOTION_TRIES)
    ur5.gripper.open_gripper()
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="[mug_tree, bowl_on_mug, bottle_in_box]")
    parser.add_argument("-a", "--pick-load-paths", type=str, nargs="+")
    parser.add_argument("-b", "--place-load-paths", type=str, nargs="+")

    parser.add_argument("-p", "--platform", default=False, action="store_true",
                        help="First take a point cloud of a platform. Then subtract the platform from the next point cloud.")
    parser.add_argument("-n", "--no-in-hand", default=False, action="store_true")

    main(parser.parse_args())
