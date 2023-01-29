import numpy as np
import pickle
import rospy
from scipy.spatial.transform import Rotation
import time

from online_isec import constants
from online_isec.point_cloud_proxy import RealsenseStructurePointCloudProxy
import online_isec.utils as isec_utils
import online_isec.utils as isec_utils
from online_isec import perception
from online_isec.ur5 import UR5
import utils
from pybullet_planning.pybullet_tools import utils as pu


def pick(mug_pc_complete, mug_param, ur5, safe_release: bool=False):

    with open("data/real_pick_clone.pkl", "rb") as f:
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


def main():

    rospy.init_node("moveit_plan_pick_place")
    pc_proxy = RealsenseStructurePointCloudProxy()

    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    cloud = pc_proxy.get_all()
    assert cloud is not None

    mug_pc_complete, mug_param, tree_pc_complete, tree_param, canon_mug, canon_tree = perception.mug_tree_perception(
        pc_proxy, np.array(constants.DESK_CENTER), ur5.tf_proxy, ur5.moveit_scene,
        add_mug_to_planning_scene=False, add_tree_to_planning_scene=False, rviz_pub=ur5.rviz_pub,
        mug_save_decomposition=True, close_proxy=True
    )

    T_m_to_g = pick(mug_pc_complete, mug_param, ur5)

    with open("data/real_place_clone.pkl", "rb") as f:
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
    print("Best fit spatial transform:")
    print(T_new_m_to_t)

    T_t_to_b = utils.pos_rot_to_transform(tree_param[0] + constants.DESK_CENTER, utils.yaw_to_rot(tree_param[1]))
    T_new_m_to_b = np.matmul(T_t_to_b, T_new_m_to_t)

    T_g_to_b = np.matmul(T_new_m_to_b, np.linalg.inv(T_m_to_g))

    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_g_to_b))
    input("Release?")
    ur5.gripper.open_gripper()

    input("Reset?")
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


if __name__ == "__main__":
    main()
