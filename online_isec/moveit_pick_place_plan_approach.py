import argparse
import numpy as np
from numpy.typing import NDArray
import os
import pickle
import rospy
from scipy.spatial.transform import Rotation
from typing import Any, Dict, Tuple
import time

from online_isec import constants
from online_isec import perception
import online_isec.utils as isec_utils
from online_isec.point_cloud_proxy_sync import RealsenseStructurePointCloudProxy
from online_isec.ur5 import UR5
from online_isec.simulation import Simulation
import utils


def pick(
    mug_pc_complete: NDArray, mug_param: Tuple[NDArray, NDArray, NDArray],
    ur5: UR5, load_path: str, safe_release: bool=False, add_mug_to_scene: bool=False) -> NDArray:

    with open(load_path, "rb") as f:
        data = pickle.load(f)
        index, target_quat = data["index"], data["quat"]
        target_pos = mug_pc_complete[index]

    target_pos = target_pos + constants.DESK_CENTER
    target_rot = np.matmul(
        mug_param[2],
        Rotation.from_quat(target_quat).as_matrix()
    )
    target_quat = Rotation.from_matrix(target_rot).as_quat()

    T = utils.pos_quat_to_transform(target_pos, target_quat)
    T_pre = utils.pos_quat_to_transform(*utils.move_hand_back((target_pos, target_quat), -0.075))
    if safe_release:
        T_pre_safe = utils.pos_quat_to_transform(*utils.move_hand_back((target_pos, target_quat), -0.01))

    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_pre))
    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T))
    ur5.gripper.close_gripper()

    if add_mug_to_scene:
        # Add mug to the planning scene.
        pos, quat = utils.transform_to_pos_quat(
            isec_utils.desk_obj_param_to_base_link_T(mug_param[1], mug_param[2], np.array(constants.DESK_CENTER), ur5.tf_proxy))
        ur5.moveit_scene.add_object("tmp.stl", "mug", pos, quat)

    # Lock mug to flange in the moveit scene.
    ur5.moveit_scene.attach_object("mug")

    # Calcualte mug to gripper transform at the point when we grasp it.
    # TODO: If the grasp moved the mug we wouldn't know.
    gripper_pos, gripper_rot = ur5.get_end_effector_pose()
    T_g_to_b = utils.pos_quat_to_transform(gripper_pos, gripper_rot)
    T_m_to_b = utils.pos_rot_to_transform(mug_param[1] + constants.DESK_CENTER, mug_param[2])
    T_m_to_g = np.matmul(np.linalg.inv(T_g_to_b), T_m_to_b)

    if safe_release:
        ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_pre_safe))
        rospy.sleep(1)
        ur5.gripper.open_gripper()
        ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_pre))

    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_pre), num_plans=1)

    return T_m_to_g


def place(
    ur5: UR5, canon_mug: Dict[str, Any], canon_tree: Dict[str, Any],
    mug_param: Tuple[NDArray, NDArray, NDArray], tree_param: Tuple[NDArray, NDArray],
    T_m_to_g: NDArray, sim: Simulation, load_path: str):

    with open(load_path, "rb") as f:
        place_data = pickle.load(f)

    knns = place_data["knns"]
    deltas = place_data["deltas"]
    target_indices = place_data["target_indices"]
    T_g_pre_to_g = place_data["T_g_pre_to_g"]

    mug_orig = utils.canon_to_pc(canon_mug, mug_param)
    tree_orig = canon_tree["canonical_obj"]

    anchors = mug_orig[knns]
    targets_mug = np.mean(anchors + deltas, axis=1)

    targets_tree = tree_orig[target_indices]

    T_new_m_to_t, _, _ = utils.best_fit_transform(targets_mug, targets_tree)
    T_t_to_b = utils.pos_rot_to_transform(tree_param[0] + constants.DESK_CENTER, tree_param[1])
    T_new_m_to_b = np.matmul(T_t_to_b, T_new_m_to_t)

    # Wiggle mug out of potential collision.
    sim.remove_all_objects()
    mug_id = sim.add_object("tmp.urdf", *utils.transform_to_pos_quat(T_new_m_to_b))
    tree_id = sim.add_object("data/real_tree.urdf", *utils.transform_to_pos_quat(T_t_to_b), fixed_base=True)
    # new_m_pos, new_m_quat = sim.wiggle(mug_id, tree_id)
    # T_new_m_to_b = utils.pos_quat_to_transform(new_m_pos, new_m_quat)

    T_g_to_b = np.matmul(T_new_m_to_b, np.linalg.inv(T_m_to_g))
    T_g_pre_to_b = np.matmul(T_g_to_b, T_g_pre_to_g)

    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_g_pre_to_b))
    ur5.moveit_scene.detach_object("mug")
    ur5.moveit_scene.remove_object("mug")

    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_g_to_b), num_plans=1)


def main(args):

    rospy.init_node("moveit_plan_pick_place_plan_approach")
    pc_proxy = RealsenseStructurePointCloudProxy()

    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)
    ur5.gripper.open_gripper(position=70)

    sim = Simulation()

    mug_pc_complete, mug_param, tree_pc_complete, tree_param, canon_mug, canon_tree, _, _ = perception.mug_tree_perception(
        pc_proxy, np.array(constants.DESK_CENTER), ur5.tf_proxy, ur5.moveit_scene,
        add_mug_to_planning_scene=not args.disable_mug_collisions_during_pick, add_tree_to_planning_scene=True, rviz_pub=ur5.rviz_pub,
        mug_save_decomposition=True, ablate_no_mug_warping=args.ablate_no_mug_warping, any_rotation=args.any_rotation,
        short_mug_platform=args.short_platform, tall_mug_platform=args.tall_platform
    )

    T_m_to_g = pick(mug_pc_complete, mug_param, ur5, args.pick_load_path + ".pkl", add_mug_to_scene=args.disable_mug_collisions_during_pick)

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
    parser.add_argument("-a", "--any-rotation", default=False, action="store_true")
    parser.add_argument("-t", "--tall-platform", default=False, action="store_true")
    parser.add_argument("-s", "--short-platform", default=False, action="store_true")
    parser.add_argument("--pick-load-path", type=str, default="data/230201_real_pick_clone", help="Postfix added automatically.")
    parser.add_argument("--place-load-path", type=str, default="data/230201_real_place_clone", help="Postfix added automatically.")
    parser.add_argument("--ablate-no-mug-warping", default=False, action="store_true")
    parser.add_argument("--disable-mug-collisions-during-pick", default=False, action="store_true")
    main(parser.parse_args())
