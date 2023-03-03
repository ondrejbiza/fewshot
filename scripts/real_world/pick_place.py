import argparse
import os
import pickle

import numpy as np
from numpy.typing import NDArray
import rospy
from scipy.spatial.transform import Rotation

from src import utils
from src.real_world import constants, perception
import src.real_world.utils as rw_utils
from src.real_world.point_cloud_proxy_sync import PointCloudProxy
from src.real_world.ur5 import UR5
from src.real_world.simulation import Simulation


def pick_simple(
    mug_pcd_complete: NDArray, mug_param: utils.ObjParam,
    ur5: UR5, load_path: str, safe_release: bool=False) -> NDArray:

    with open(load_path, "rb") as f:
        data = pickle.load(f)
        index, target_quat = data["index"], data["quat"]
        target_pos = mug_pcd_complete[index]

    target_pos = target_pos + constants.DESK_CENTER
    target_rot = np.matmul(
        Rotation.from_quat(mug_param.quat).as_matrix(),
        Rotation.from_quat(target_quat).as_matrix()
    )
    target_quat = Rotation.from_matrix(target_rot).as_quat()

    T = utils.pos_quat_to_transform(target_pos, target_quat)
    T_pre = utils.pos_quat_to_transform(*rw_utils.move_hand_back(target_pos, target_quat, -0.1))
    if safe_release:
        T_pre_safe = utils.pos_quat_to_transform(*rw_utils.move_hand_back(target_pos, target_quat, -0.01))

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(T_pre))

    # Remove mug from planning scene.
    ur5.moveit_scene.remove_object("mug")

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(T))
    ur5.gripper.close_gripper()

    # Add mug back to the planning scene.
    pos, quat = utils.transform_to_pos_quat(
        rw_utils.desk_obj_param_to_base_link_T(mug_param.position, mug_param.quat, np.array(constants.DESK_CENTER), ur5.tf_proxy))
    ur5.moveit_scene.add_object("tmp_source.stl", "mug", pos, quat)

    # Lock mug to flange in the moveit scene.
    ur5.moveit_scene.attach_object("mug")

    # Calcualte mug to gripper transform at the point when we grasp it.
    # TODO: If the grasp moved the mug we wouldn't know.
    gripper_pos, gripper_rot = ur5.get_end_effector_pose()
    T_g_to_b = utils.pos_quat_to_transform(gripper_pos, gripper_rot)
    T_m_to_b = utils.pos_quat_to_transform(mug_param.position + constants.DESK_CENTER, mug_param.quat)
    T_m_to_g = np.matmul(np.linalg.inv(T_g_to_b), T_m_to_b)

    if safe_release:
        ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(T_pre_safe))
        rospy.sleep(1)
        ur5.gripper.open_gripper()
        ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(T_pre))

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(T_pre), num_plans=1)

    return T_m_to_g


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

    trans_tool0_to_ws = trans_robotiq_to_ws @ np.linalg.inv(rw_utils.robotiq_to_tool0())
    trans_tool0_controller_to_tool0 = ur5.tf_proxy.lookup_transform("tool0_controller", "tool0")
    trans_tool0_controller_to_ws = trans_tool0_to_ws @ trans_tool0_controller_to_tool0
    trans_tool0_controller_to_b = rw_utils.workspace_to_base() @ trans_tool0_controller_to_ws

    trans_g = trans_tool0_controller_to_b
    trans_pre_g = utils.pos_quat_to_transform(*rw_utils.move_hand_back(*utils.transform_to_pos_quat(trans_g), -0.1))

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_g))

    # Remove mug from planning scene.
    ur5.moveit_scene.remove_object("mug")

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_g))
    ur5.gripper.close_gripper()

    # Add mug back to the planning scene.
    pos, quat = utils.transform_to_pos_quat(
        rw_utils.desk_obj_param_to_base_link_T(source_param.position, source_param.quat, np.array(constants.DESK_CENTER), ur5.tf_proxy))
    ur5.moveit_scene.add_object("tmp_source.stl", "mug", pos, quat)

    # Lock mug to flange in the moveit scene.
    ur5.moveit_scene.attach_object("mug")

    # Calcualte mug to gripper transform at the point when we grasp it.
    # TODO: If the grasp moved the mug we wouldn't know.
    gripper_pos, gripper_rot = ur5.get_end_effector_pose()
    T_g_to_b = utils.pos_quat_to_transform(gripper_pos, gripper_rot)
    T_m_to_b = utils.pos_quat_to_transform(source_param.position + constants.DESK_CENTER, source_param.quat)
    T_m_to_g = np.matmul(np.linalg.inv(T_g_to_b), T_m_to_b)

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_g), num_plans=1)

    return T_m_to_g


def place(
    ur5: UR5, canon_mug: utils.CanonObj, canon_tree: utils.CanonObj,
    mug_param: utils.ObjParam, tree_param: utils.ObjParam,
    T_m_to_g: NDArray, sim: Simulation, load_path: str):

    with open(load_path, "rb") as f:
        place_data = pickle.load(f)

    knns = place_data["knns"]
    deltas = place_data["deltas"]
    target_indices = place_data["target_indices"]
    T_g_pre_to_g = place_data["T_g_pre_to_g"]

    mug_orig = canon_mug.to_pcd(mug_param)
    tree_orig = canon_tree.to_pcd(tree_param)

    anchors = mug_orig[knns]
    targets_mug = np.mean(anchors + deltas, axis=1)

    targets_tree = tree_orig[target_indices]

    T_new_m_to_t, _, _ = utils.best_fit_transform(targets_mug, targets_tree)
    T_t_to_b = utils.pos_quat_to_transform(tree_param.position + constants.DESK_CENTER, tree_param.quat)
    T_new_m_to_b = np.matmul(T_t_to_b, T_new_m_to_t)

    # Wiggle mug out of potential collision.
    sim.remove_all_objects()
    mug_id = sim.add_object("tmp_source.urdf", *utils.transform_to_pos_quat(T_new_m_to_b))
    tree_id = sim.add_object("tmp_target.urdf", *utils.transform_to_pos_quat(T_t_to_b), fixed_base=True)
    # new_m_pos, new_m_quat = sim.wiggle(mug_id, tree_id)
    # T_new_m_to_b = utils.pos_quat_to_transform(new_m_pos, new_m_quat)

    T_g_to_b = np.matmul(T_new_m_to_b, np.linalg.inv(T_m_to_g))
    T_g_pre_to_b = np.matmul(T_g_to_b, T_g_pre_to_g)

    # Remove mug from the planning scene.
    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(T_g_pre_to_b))
    ur5.moveit_scene.detach_object("mug")
    ur5.moveit_scene.remove_object("mug")

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(T_g_to_b), num_plans=1)


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
    mug_pcd_complete, mug_param, tree_pcd_complete, tree_param, canon_mug, canon_tree, _, _ = out

    if args.pick_contacts:
        T_m_to_g = pick_contacts(ur5, canon_mug, mug_param, args.pick_load_path + ".pkl")
    else:
        T_m_to_g = pick_simple(mug_pcd_complete, mug_param, ur5, args.pick_load_path + ".pkl")

    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    load_path = args.place_load_path + ".pkl"
    if not os.path.isfile(load_path):
        print("Place clone file not found.")
    else:
        place(ur5, canon_mug, canon_tree, mug_param, tree_param, T_m_to_g, sim, load_path)

    # input("Release?")
    ur5.gripper.open_gripper()

    # input("Reset?")
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="[mug_tree, bowl_on_mug, bottle_in_box]")
    parser.add_argument("pick_load_path", type=str, help="Postfix added automatically.")
    parser.add_argument("place_load_path", type=str, help="Postfix added automatically.")

    parser.add_argument("-a", "--any-rotation", default=False, action="store_true",
                        help="Try to determine SE(3) object pose. Otherwise, determine a planar pose.")
    parser.add_argument("-c", "--pick-contacts", default=False, action="store_true")
    parser.add_argument("-p", "--platform", default=False, action="store_true",
                        help="First take a point cloud of a platform. Then subtract the platform from the next point cloud.")

    parser.add_argument("--ablate-no-mug-warping", default=False, action="store_true")

    main(parser.parse_args())
