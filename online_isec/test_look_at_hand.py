import numpy as np
import pybullet as pb
import rospy
from scipy.spatial.transform import Rotation
import time
import pickle

from online_isec.ur5 import UR5
import online_isec.utils as isec_utils
from pybullet_planning.pybullet_tools import utils as pu
from online_isec import constants
import utils
from online_isec.point_cloud_proxy import RealsenseStructurePointCloudProxy
import viz_utils
from online_isec import perception


def main():

    rospy.init_node("test_robotiq_in_pybullet")

    pc_proxy = RealsenseStructurePointCloudProxy()

    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    mug_pc_complete, mug_param, tree_pc_complete, tree_param = perception.mug_tree_perception(
        pc_proxy, np.array(constants.DESK_CENTER), ur5.tf_proxy, ur5.moveit_scene,
        add_mug_to_planning_scene=False, add_tree_to_planning_scene=True, rviz_pub=ur5.rviz_pub
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
    T_pre = utils.pos_quat_to_transform(*utils.move_hand_back((target_pos, target_quat), -0.05))

    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_pre))
    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T))
    ur5.gripper.close_gripper()
    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T_pre))

    pu.connect(use_gui=True, show_sliders=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    robotiq = pb.loadURDF("data/robotiq.urdf", useFixedBase=True)
    pu.set_pose(robotiq, (np.array([0., 0, 0.]), np.array([1., 0., 0., 0.])))

    joint_name = "robotiq_85_left_knuckle_joint"
    joint_index = None
    for idx in pu.get_joints(robotiq):
        if pu.get_joint_name(robotiq, idx) == joint_name:
            joint_index = idx
            break
    assert joint_index is not None

    pos_ws = np.array([0.16, -0.16, 0.2])
    pos = pos_ws + constants.DESK_CENTER
    T = utils.pos_quat_to_transform(pos, np.array([0., 0., 0., 1.]))

    # input("red button 1")
    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T))
    time.sleep(0.5)
    cloud1 = pc_proxy.get_all()

    rot1 = Rotation.from_euler("z", (2 * np.pi) / 3.).as_matrix()
    rot2 = Rotation.from_euler("z", (4 * np.pi) / 3.).as_matrix()

    T2 = np.copy(T)
    T2[:3, :3] = np.matmul(rot1, T2[:3, :3])
    # input("red button 2")
    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T2))
    time.sleep(0.5)
    cloud2 = pc_proxy.get_all()

    T3 = np.copy(T)
    T3[:3, :3] = np.matmul(rot2, T3[:3, :3])
    # input("red button 3")
    ur5.plan_and_execute_pose_target_2(*utils.transform_to_pos_quat(T3))
    time.sleep(0.5)
    cloud3 = pc_proxy.get_all()    

    cloud1 = cloud1 - constants.DESK_CENTER
    cloud2 = cloud2 - constants.DESK_CENTER
    cloud3 = cloud3 - constants.DESK_CENTER

    box_min = [-0.1, -0.1, -0.15]
    box_max = [0.1, 0.1, 0.05]

    # recenter PCs on tool
    cloud2 = cloud2 - pos_ws
    cloud3 = cloud3 - pos_ws

    cloud1 = cloud1[np.logical_and(cloud1[:, 0] >= box_min[0], cloud1[:, 0] <= box_max[0])]
    cloud1 = cloud1[np.logical_and(cloud1[:, 1] >= box_min[1], cloud1[:, 1] <= box_max[1])]
    cloud1 = cloud1[np.logical_and(cloud1[:, 2] >= box_min[2], cloud1[:, 2] <= box_max[2])]

    cloud2 = cloud2[np.logical_and(cloud2[:, 0] >= box_min[0], cloud2[:, 0] <= box_max[0])]
    cloud2 = cloud2[np.logical_and(cloud2[:, 1] >= box_min[1], cloud2[:, 1] <= box_max[1])]
    cloud2 = cloud2[np.logical_and(cloud2[:, 2] >= box_min[2], cloud2[:, 2] <= box_max[2])]

    cloud3 = cloud3[np.logical_and(cloud3[:, 0] >= box_min[0], cloud3[:, 0] <= box_max[0])]
    cloud3 = cloud3[np.logical_and(cloud3[:, 1] >= box_min[1], cloud3[:, 1] <= box_max[1])]
    cloud3 = cloud3[np.logical_and(cloud3[:, 2] >= box_min[2], cloud3[:, 2] <= box_max[2])]

    # rotate PCs
    cloud2 = np.matmul(cloud2, np.linalg.inv(rot1).T)  # TODO: inv and .T cancel out
    cloud3 = np.matmul(cloud3, np.linalg.inv(rot2).T)
    # recenter PCs on workspace center
    cloud2 = cloud2 + pos_ws
    cloud3 = cloud3 + pos_ws

    c = np.concatenate([cloud1, cloud2, cloud3])
    viz_utils.o3d_visualize(utils.create_o3d_pointcloud(c))

    # while not rospy.is_shutdown():
    #     time.sleep(0.1)
    #     fract = ur5.gripper.get_open_fraction()
    #     pu.set_joint_positions(robotiq, [0, 2, 4, 5, 6, 7], [fract, fract, fract, -fract, fract, -fract])


main()
