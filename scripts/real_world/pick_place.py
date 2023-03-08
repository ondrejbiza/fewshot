import argparse
import os
import pickle

import numpy as np
from numpy.typing import NDArray
import rospy
import torch

from src import object_warping, utils, viz_utils
from src.real_world import constants, perception
import src.real_world.utils as rw_utils
from src.real_world.point_cloud_proxy import PointCloudProxy
from src.real_world.ur5 import UR5
from src.real_world.simulation import Simulation


def pick_simple(source_pcd_complete: NDArray, source_param: utils.ObjParam, 
                ur5: UR5, load_path: str) -> NDArray:
    """Pick object by following a single vertex position and a pre-recorded pose."""
    with open(load_path, "rb") as f:
        data = pickle.load(f)
        index, t0_tip_quat = data["index"], data["quat"]
        t0_tip_pos_in_ws = source_pcd_complete[index]

    t0_tip_pos_in_base = t0_tip_pos_in_ws + constants.DESK_CENTER
    t0_tip_rot = np.matmul(
        utils.quat_to_rotm(source_param.quat),
        utils.quat_to_rotm(t0_tip_quat)
    )
    t0_tip_quat = utils.rotm_to_quat(t0_tip_rot)

    trans_t0_tip_to_b = utils.pos_quat_to_transform(t0_tip_pos_in_base, t0_tip_quat)
    trans_t0_to_b = trans_t0_tip_to_b @ np.linalg.inv(rw_utils.tool0_tip_to_tool0())

    trans_pre_t0_to_b = utils.pos_quat_to_transform(*rw_utils.move_hand_back(*utils.transform_to_pos_quat(trans_t0_to_b), -0.0))

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_t0_to_b))

    # Remove source object from planning scene.
    ur5.moveit_scene.remove_object("source")

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_t0_to_b))
    ur5.gripper.close_gripper()

    # Add source object back to the planning scene.
    pos, quat = utils.transform_to_pos_quat(
        rw_utils.desk_obj_param_to_base_link_T(source_param.position, source_param.quat, np.array(constants.DESK_CENTER), ur5.tf_proxy))
    ur5.moveit_scene.add_object("tmp_source.stl", "source", pos, quat)

    # Lock source object to flange in the moveit scene.
    ur5.moveit_scene.attach_object("source")

    # Calculate source object to gripper transform at the point when we grasp it.
    trans_t0_to_b = utils.pos_quat_to_transform(*ur5.get_tool0_to_base())
    trans_source_to_ws = source_param.get_transform()
    trans_source_to_b = rw_utils.workspace_to_base() @ trans_source_to_ws
    trans_source_to_t0 = np.linalg.inv(trans_t0_to_b) @ trans_source_to_b

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_t0_to_b), num_plans=1)

    return trans_source_to_t0


def pick_contacts(ur5: UR5, canon_source: utils.CanonObj, source_param: utils.ObjParam, load_path: str):

    with open(load_path, "rb") as f:
        d = pickle.load(f)
    index = d["index"]
    pos_robotiq = d["pos_robotiq"]

    source_pcd_complete = canon_source.to_pcd(source_param)
    source_points = source_pcd_complete[index]

    # Pick pose in canonical frame..
    trans, _, _ = utils.best_fit_transform(pos_robotiq, source_points)
    # Pick pose in workspace frame.
    trans_robotiq_to_ws = source_param.get_transform() @ trans

    trans_t0_to_ws = trans_robotiq_to_ws @ np.linalg.inv(rw_utils.robotiq_to_tool0())
    trans_t0_to_b = rw_utils.workspace_to_base() @ trans_t0_to_ws

    trans_g = trans_t0_to_b
    trans_pre_g = utils.pos_quat_to_transform(*rw_utils.move_hand_back(*utils.transform_to_pos_quat(trans_g), -0.1))

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_g))

    # Remove mug from planning scene.
    ur5.moveit_scene.remove_object("source")

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_g))
    ur5.gripper.close_gripper()

    # Add mug back to the planning scene.
    pos, quat = utils.transform_to_pos_quat(
        rw_utils.desk_obj_param_to_base_link_T(source_param.position, source_param.quat, np.array(constants.DESK_CENTER), ur5.tf_proxy))
    ur5.moveit_scene.add_object("tmp_source.stl", "source", pos, quat)

    # Lock mug to flange in the moveit scene.
    ur5.moveit_scene.attach_object("source")

    # Calcualte mug to gripper transform at the point when we grasp it.
    trans_t0_to_b = utils.pos_quat_to_transform(*ur5.get_tool0_to_base())
    trans_source_to_ws = source_param.get_transform()
    trans_source_to_b = rw_utils.workspace_to_base() @ trans_source_to_ws
    trans_source_to_t0 = np.linalg.inv(trans_t0_to_b) @ trans_source_to_b

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_g), num_plans=1)

    return trans_source_to_t0


def place(
    ur5: UR5, canon_source: utils.CanonObj, canon_target: utils.CanonObj,
    source_param: utils.ObjParam, target_param: utils.ObjParam,
    trans_source_to_t0: NDArray, sim: Simulation, load_path: str):

    with open(load_path, "rb") as f:
        place_data = pickle.load(f)

    knns = place_data["knns"]
    deltas = place_data["deltas"]
    target_indices = place_data["target_indices"]
    T_g_pre_to_g = place_data["T_g_pre_to_g"]

    source_orig = canon_source.to_pcd(source_param)
    target_orig = canon_target.to_pcd(target_param)

    anchors = source_orig[knns]
    contacts_source = np.mean(anchors + deltas, axis=1)
    contacts_target = target_orig[target_indices]

    trans_new_source_to_target, _, _ = utils.best_fit_transform(contacts_source, contacts_target)
    trans_target_to_b = utils.pos_quat_to_transform(target_param.position + constants.DESK_CENTER, target_param.quat)
    trans_new_source_to_b = np.matmul(trans_target_to_b, trans_new_source_to_target)

    # Wiggle mug out of potential collision.
    # sim.remove_all_objects()
    # source_id = sim.add_object("tmp_source.urdf", *utils.transform_to_pos_quat(trans_new_source_to_b))
    # target_id = sim.add_object("tmp_target.urdf", *utils.transform_to_pos_quat(trans_target_to_b), fixed_base=True)
    # new_m_pos, new_m_quat = sim.wiggle(mug_id, tree_id)
    # T_new_m_to_b = utils.pos_quat_to_transform(new_m_pos, new_m_quat)

    trans_g_to_b = np.matmul(trans_new_source_to_b, np.linalg.inv(trans_source_to_t0))
    trans_g_pre_to_b = np.matmul(trans_g_to_b, T_g_pre_to_g)

    # Remove mug from the planning scene.
    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_g_pre_to_b))
    ur5.moveit_scene.detach_object("source")
    ur5.moveit_scene.remove_object("source")

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_g_to_b), num_plans=1)


def main(args):

    rospy.init_node("pick_place")
    pc_proxy = PointCloudProxy()

    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)
    ur5.gripper.open_gripper(position=70)

    platform_pcd = None
    if args.platform:
        cloud = pc_proxy.get_all()
        assert cloud is not None
        platform_pcd = perception.platform_segmentation(cloud)
        input("Platform captured. Continue? ")

    cloud = pc_proxy.get_all()
    assert cloud is not None

    sim = Simulation()

    if args.task == "mug_tree":
        canon_source = utils.CanonObj.from_pickle("data/230227_ndf_mugs_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_target = utils.CanonObj.from_pickle("data/230228_simple_trees_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_source.init_scale = 0.7
        canon_target.init_scale = 1.
        source_pcd, target_pcd = perception.mug_tree_segmentation(cloud, platform_pcd=platform_pcd)
    elif args.task == "bowl_on_mug":
        canon_source = utils.CanonObj.from_pickle("data/230227_ndf_bowls_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_target = utils.CanonObj.from_pickle("data/230227_ndf_mugs_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_source.init_scale = 0.8
        canon_target.init_scale = 0.7
        source_pcd, target_pcd = perception.bowl_mug_segmentation(cloud, platform_pcd=platform_pcd)
    elif args.task == "bottle_in_box":
        canon_source = utils.CanonObj.from_pickle("data/230227_ndf_bottles_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_target = utils.CanonObj.from_pickle("data/230228_boxes_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_source.init_scale = 1.
        canon_target.init_scale = 1.
        source_pcd, target_pcd = perception.bottle_box_segmentation(cloud, platform_pcd=platform_pcd)
    else:
        raise ValueError("Unknown task.")

    out = perception.warping(
        source_pcd, target_pcd, canon_source, canon_target, source_any_rotation=args.any_rotation,
        tf_proxy=ur5.tf_proxy, moveit_scene=ur5.moveit_scene, source_save_decomposition=True,
        target_save_decomposition=True, add_source_to_planning_scene=True, add_target_to_planning_scene=True,
        rviz_pub=ur5.rviz_pub, grow_source_object=True
    )
    source_pcd_complete, source_param, target_pcd_complete, target_param = out

    if args.pick_contacts:
        trans_source_to_t0 = pick_contacts(ur5, canon_source, source_param, args.pick_load_path)
    else:
        trans_source_to_t0 = pick_simple(source_pcd_complete, source_param, ur5, args.pick_load_path)

    # Take an in-hand image.
    if args.platform:
        input("Platform removed?")

    cloud = pc_proxy.get_all()
    assert cloud is not None

    robotiq_id = sim.add_object("data/robotiq.urdf", np.array([0., 0., 0.]), np.array([0., 0., 0., 1.]))
    source_param, source_pcd_complete, trans_source_to_t0, in_hand_pcd = perception.reestimate_tool0_to_source(
        cloud, ur5, robotiq_id, sim, canon_source, source_param, trans_source_to_t0)

    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    load_path = args.place_load_path + ".pkl"
    if not os.path.isfile(load_path):
        print("Place clone file not found.")
    else:
        place(ur5, canon_source, canon_target, source_param, target_param, trans_source_to_t0, sim, load_path)

    # input("Release?")
    ur5.gripper.open_gripper()

    # input("Reset?")
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="[mug_tree, bowl_on_mug, bottle_in_box]")
    parser.add_argument("pick_load_path", type=str)
    parser.add_argument("place_load_path", type=str)

    parser.add_argument("-a", "--any-rotation", default=False, action="store_true",
                        help="Try to determine SE(3) object pose. Otherwise, determine a planar pose.")
    parser.add_argument("-c", "--pick-contacts", default=False, action="store_true")
    parser.add_argument("-p", "--platform", default=False, action="store_true",
                        help="First take a point cloud of a platform. Then subtract the platform from the next point cloud.")

    parser.add_argument("--ablate-no-mug-warping", default=False, action="store_true")

    main(parser.parse_args())
