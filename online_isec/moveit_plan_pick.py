import numpy as np
import pickle
import rospy
from scipy.spatial.transform import Rotation

from online_isec import constants
from online_isec.point_cloud_proxy import RealsenseStructurePointCloudProxy
import online_isec.utils as isec_utils
import online_isec.utils as isec_utils
from online_isec import perception
from online_isec.ur5 import UR5
import utils


def main():

    rospy.init_node("easy_perception")
    pc_proxy = RealsenseStructurePointCloudProxy()
    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    cloud = pc_proxy.get_all()
    assert cloud is not None

    mug_pc_complete, mug_param, tree_pc_complete, tree_param = perception.mug_tree_perception(
        pc_proxy, np.array(constants.DESK_CENTER), ur5.tf_proxy, ur5.moveit_scene
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

    print("base to tool0_controller")
    print("Target pos: {}".format(target_pos))
    print("Target quat: {}".format(target_quat))

    T = utils.pos_quat_to_transform(target_pos, target_quat)
    T_pre = utils.pos_quat_to_transform(*utils.move_hand_back((target_pos, target_quat), -0.05))

    print(T_pre)
    print(T)
    
    T = isec_utils.base_tool0_controller_to_base_link_flange(T, ur5.tf_proxy)
    T_pre = isec_utils.base_tool0_controller_to_base_link_flange(T_pre, ur5.tf_proxy)

    print(T_pre)
    print(T)

    input("big red button")
    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(T_pre))

    input("big red button part 2")
    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(T))

    input("reset")
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)



main()
