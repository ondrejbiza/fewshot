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
from online_isec.rviz_pub import MeshViz
import utils
from pybullet_planning.pybullet_tools import utils as pu


def main():

    rospy.init_node("easy_perception")
    pc_proxy = RealsenseStructurePointCloudProxy()
    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    cloud = pc_proxy.get_all()
    assert cloud is not None

    mesh_viz = MeshViz()

    mug_pc_complete, mug_param, tree_pc_complete, tree_param = perception.mug_tree_perception(
        pc_proxy, np.array(constants.DESK_CENTER), ur5.tf_proxy, ur5.moveit_scene,
        add_mug_to_planning_scene=True, add_tree_to_planning_scene=True, mesh_viz=mesh_viz
    )

    with open("data/real_pick_clone.pkl", "rb") as f:
        data = pickle.load(f)
        index, target_quat = data["index"], data["quat"]
        target_pos = mug_pc_complete[index]

    target_pos = target_pos + constants.DESK_CENTER
    tmp = np.matmul(
        utils.yaw_to_rot(mug_param[2]),
        Rotation.from_quat(target_quat).as_matrix()
    )

    target_quat = Rotation.from_matrix(tmp).as_quat()

    T = utils.pos_quat_to_transform(target_pos, target_quat)
    T_pre = utils.pos_quat_to_transform(*utils.move_hand_back((target_pos, target_quat), 0.05))

    T = isec_utils.base_tool0_controller_to_base_link_flange(T, ur5.tf_proxy)
    T_pre = isec_utils.base_tool0_controller_to_base_link_flange(T_pre, ur5.tf_proxy)

    input("big red button")
    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(T_pre))

    input("big red button part 2")
    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(T))

    with open("data/real_place_clone.pkl", "rb") as f:
        place_data = pickle.load(f)

    knns = place_data["knns"]
    deltas = place_data["deltas"]
    target_indices = place_data["target_indices"]

    rot = utils.yaw_to_rot(mug_param[2])
    new_deltas = np.matmul(deltas, rot.T)

    anchors = mug_pc_complete[knns]
    targets = np.mean(anchors + new_deltas, axis=1)

    points_2 = tree_pc_complete[target_indices]

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2])
    # ax.scatter(points_2[:, 0], points_2[:, 1], points_2[:, 2])
    # plt.show()

    vp_to_p2, _, _ = utils.best_fit_transform(targets + constants.DESK_CENTER, points_2 + constants.DESK_CENTER)
    print("Best fit spatial transform:")
    print(vp_to_p2)

    # wiggle

    # setup simulated scene
    pu.connect(use_gui=True, show_sliders=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    T = utils.pos_rot_to_transform(mug_param[1] + constants.DESK_CENTER, utils.yaw_to_rot(mug_param[2]))
    T = np.matmul(vp_to_p2, T)
    pu.load_model("../tmp.obj", utils.transform_to_pos_quat(T))

    T = utils.pos_rot_to_transform(tree_param[0] + constants.DESK_CENTER, utils.yaw_to_rot(tree_param[1]))
    pu.load_model("../data/real_tree2.obj", utils.transform_to_pos_quat(T))

    while True:
        time.sleep(1)

    # mug_T = utils.pos_quat_to_transform(mug_param[1], utils.yaw_to_rot(mug_param[2]))
    gripper_pos, gripper_quat = ur5.get_end_effector_pose()
    hand_T = utils.pos_quat_to_transform(gripper_pos, gripper_quat)

    # g_t_m = np.matmul(np.linalg.inv(hand_T), mug_T)
    # g_t_m_pos, g_t_m_quat = utils.transform_to_pos_quat(g_t_m)

    new_hand_T = np.matmul(vp_to_p2, hand_T)

    T = isec_utils.base_tool0_controller_to_base_link_flange(new_hand_T, ur5.tf_proxy)

    input("big red button part 3")
    ur5.gripper.close_gripper()

    isec_utils.attach_obj_to_hand("mug", ur5.moveit_scene)
    isec_utils.check_added_to_moveit_scene("mug", ur5.moveit_scene, obj_is_attached=True)

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(T))
    ur5.gripper.open_gripper()

    input("reset")
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


main()
