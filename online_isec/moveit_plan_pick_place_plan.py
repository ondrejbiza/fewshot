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

    if safe_release:
        ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_pre_safe))
        rospy.sleep(1)
        ur5.gripper.open_gripper()
        ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_pre))

    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_pre))


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

    pick(mug_pc_complete, mug_param, ur5)

    with open("data/real_place_clone.pkl", "rb") as f:
        place_data = pickle.load(f)

    knns = place_data["knns"]
    deltas = place_data["deltas"]
    target_indices = place_data["target_indices"]

    mug_rot = utils.yaw_to_rot(mug_param[2])
    deltas_rot = np.matmul(deltas, mug_rot.T)

    anchors = mug_pc_complete[knns]
    targets_mug = np.mean(anchors + deltas_rot, axis=1)

    targets_tree = tree_pc_complete[target_indices]

    rel_pose, _, _ = utils.best_fit_transform(targets_mug, targets_tree)
    print("Best fit spatial transform:")
    print(rel_pose)

    # setup simulated scene
    pu.connect(use_gui=True, show_sliders=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    T_m_to_b = utils.pos_rot_to_transform(mug_param[1], utils.yaw_to_rot(mug_param[2]))
    T_task = np.matmul(rel_pose, T_m_to_b)  # TODO: Not sure about this matmul.
    pu.load_model("../tmp.obj", utils.transform_to_pos_quat(T_task))

    T_t_to_b = utils.pos_rot_to_transform(tree_param[0], utils.yaw_to_rot(tree_param[1]))
    pu.load_model("../data/real_tree2.obj", utils.transform_to_pos_quat(T_t_to_b))

    # TODO: wiggle

    # mug_T = utils.pos_quat_to_transform(mug_param[1], utils.yaw_to_rot(mug_param[2]))
    gripper_pos, gripper_quat = ur5.get_end_effector_pose()
    T_g_to_b = utils.pos_quat_to_transform(gripper_pos, gripper_quat)

    # g_t_m = np.matmul(np.linalg.inv(hand_T), mug_T)
    # g_t_m_pos, g_t_m_quat = utils.transform_to_pos_quat(g_t_m)

    T_g_task_to_b = np.matmul(T_task, T_g_to_b)

    # isec_utils.attach_obj_to_hand("mug", ur5.moveit_scene)
    # isec_utils.check_added_to_moveit_scene("mug", ur5.moveit_scene, obj_is_attached=True)

    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_g_task_to_b))
    ur5.gripper.open_gripper()

    input("reset")
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


if __name__ == "__main__":
    main()
