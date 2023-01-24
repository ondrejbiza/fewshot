import argparse
import rospy
import time
import numpy as np
from scipy.spatial.transform import Rotation

from online_isec.ur5 import UR5
import utils
import online_isec.utils as isec_utils


def main(args):
    rospy.init_node("ur5_control")
    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    pos, quat = ur5.get_end_effector_pose()
    rotm = Rotation.from_euler("z", args.angle).as_matrix()
    rot = np.matmul(rotm, Rotation.from_quat(quat).as_matrix())
    quat = Rotation.from_matrix(rot).as_quat()

    T = utils.pos_quat_to_transform(pos, quat)
    print("base to tool0_controller")
    print(T)
    T = isec_utils.base_tool0_controller_to_base_link_flange(T, ur5.tf_proxy)
    print("base_link to flange")
    print(T)
    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(T))


parser = argparse.ArgumentParser()
parser.add_argument("angle", type=float, default=0.)
main(parser.parse_args())
