import numpy as np
import pybullet as pb
import rospy
import time

from online_isec.ur5 import UR5
from pybullet_planning.pybullet_tools import utils as pu


def main():

    rospy.init_node("test_robotiq_in_pybullet")
    ur5 = UR5()

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

    while not rospy.is_shutdown():
        time.sleep(0.1)
        fract = ur5.gripper.get_open_fraction()
        pu.set_joint_positions(robotiq, [0, 2, 4, 5, 6, 7], [fract, fract, fract, -fract, fract, -fract])


main()
