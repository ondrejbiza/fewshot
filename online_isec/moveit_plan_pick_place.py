import argparse
import numpy as np
from numpy.typing import NDArray
import os
import pickle
import rospy
from scipy.spatial.transform import Rotation
from typing import Any, Dict, Tuple

from online_isec import constants
from online_isec import perception
from online_isec.point_cloud_proxy import RealsenseStructurePointCloudProxy
from online_isec.ur5 import UR5
import utils


def pick(
    mug_pc_complete: NDArray, mug_param: Tuple[NDArray, NDArray, NDArray],
    ur5: UR5, load_path: str, safe_release: bool=False) -> NDArray:

    with open(load_path, "rb") as f:
        data = pickle.load(f)
        index, target_quat = data["index"], data["quat"]
        target_pos = mug_pc_complete[index]

    target_pos = target_pos + constants.DESK_CENTER
    target_rot = np.matmul(
        utils.yaw_to_rot(mug_param[2]),
        Rotation.from_quat(target_quat).as_matrix()
    )
    target_quat = Rotation.from_matrix(target_rot).as_quat()

    T = utils.pos_quat_to_transform(target_pos, target_quat)
    T_pre = utils.pos_quat_to_transform(*utils.move_hand_back((target_pos, target_quat), -0.05))
    if safe_release:
        T_pre_safe = utils.pos_quat_to_transform(*utils.move_hand_back((target_pos, target_quat), -0.01))

    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_pre))
    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T))
    ur5.gripper.close_gripper()

    # Calcualte mug to gripper transform at the point when we grasp it.
    # TODO: If the grasp moved the mug we wouldn't know.
    gripper_pos, gripper_rot = ur5.get_end_effector_pose()
    T_g_to_b = utils.pos_quat_to_transform(gripper_pos, gripper_rot)
    T_m_to_b = utils.pos_rot_to_transform(mug_param[1] + constants.DESK_CENTER, utils.yaw_to_rot(mug_param[2]))
    T_m_to_g = np.matmul(np.linalg.inv(T_g_to_b), T_m_to_b)

    if safe_release:
        ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_pre_safe))
        rospy.sleep(1)
        ur5.gripper.open_gripper()
        ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_pre))

    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_pre))

    return T_m_to_g


def place(
    ur5: UR5, canon_mug: Dict[str, Any], canon_tree: Dict[str, Any],
    mug_param: Tuple[NDArray, NDArray, NDArray], tree_param: Tuple[NDArray, NDArray],
    T_m_to_g: NDArray, load_path: str):

    with open(load_path, "rb") as f:
        place_data = pickle.load(f)

    knns = place_data["knns"]
    deltas = place_data["deltas"]
    target_indices = place_data["target_indices"]

    mug_orig = utils.canon_to_pc(canon_mug, mug_param)
    tree_orig = canon_tree["canonical_obj"]

    anchors = mug_orig[knns]
    targets_mug = np.mean(anchors + deltas, axis=1)

    targets_tree = tree_orig[target_indices]

    T_new_m_to_t, _, _ = utils.best_fit_transform(targets_mug, targets_tree)
    T_t_to_b = utils.pos_rot_to_transform(tree_param[0] + constants.DESK_CENTER, utils.yaw_to_rot(tree_param[1]))
    T_new_m_to_b = np.matmul(T_t_to_b, T_new_m_to_t)
    T_g_to_b = np.matmul(T_new_m_to_b, np.linalg.inv(T_m_to_g))

    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_g_to_b))


def main(args):

    rospy.init_node("moveit_plan_pick_place")
    pc_proxy = RealsenseStructurePointCloudProxy()

    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    cloud = pc_proxy.get_all()
    assert cloud is not None

    mug_pc_complete, mug_param, tree_pc_complete, tree_param, canon_mug, canon_tree = perception.mug_tree_perception(
        pc_proxy, np.array(constants.DESK_CENTER), ur5.tf_proxy, ur5.moveit_scene,
        add_mug_to_planning_scene=False, add_tree_to_planning_scene=True, rviz_pub=ur5.rviz_pub,
        mug_save_decomposition=True, close_proxy=True
    )

    T_m_to_g = pick(mug_pc_complete, mug_param, ur5, args.pick_load_path + ".pkl")
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    if args.multiple_waypoints:
        i = 0
        while True:
            load_path = args.place_load_path + "_{:d}.pkl".format(i)
            if not os.path.isfile(load_path):
                break
            place(ur5, canon_mug, canon_tree, mug_param, tree_param, T_m_to_g, load_path)
            i += 1
    else:
        load_path = args.place_load_path + ".pkl"
        if not os.path.isfile(load_path):
            print("Place clone file not found. Set -m?")
        else:
            place(ur5, canon_mug, canon_tree, mug_param, tree_param, T_m_to_g, load_path)

    # input("Release?")
    ur5.gripper.open_gripper()

    # input("Reset?")
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick-load-path", type=str, default="data/220129_real_pick_clone", help="Postfix added automatically.")
    parser.add_argument("--place-load-path", type=str, default="data/220129_real_place_clone", help="Postfix added automatically.")
    parser.add_argument("-m", "--multiple-waypoints", default=False, action="store_true", help="Multiple place waypoint.")
    main(parser.parse_args())