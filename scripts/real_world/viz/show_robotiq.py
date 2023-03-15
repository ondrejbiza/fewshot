import time

import numpy as np
import rospy

from src import utils
import src.real_world.utils as rw_utils
from src.real_world.ur5 import UR5
from src.real_world.simulation import Simulation


def main():

    rospy.init_node("save_demo")

    ur5 = UR5(setup_planning=True)
    sim = Simulation()

    robotiq_id = sim.add_object("data/robotiq.urdf", np.array([0., 0., 0.]), np.array([0., 0., 0., 1.]))
    trans_robotiq_to_tool0 = rw_utils.robotiq_to_tool0()

    while not rospy.is_shutdown():

        gripper_pos, gripper_quat = ur5.get_tool0_to_base()
        trans_t0_to_b = utils.pos_quat_to_transform(gripper_pos, gripper_quat)
        trans = trans_t0_to_b @ trans_robotiq_to_tool0
        gripper_pos, gripper_quat = utils.transform_to_pos_quat(trans)
        utils.pb_set_pose(robotiq_id, gripper_pos, gripper_quat)

        fract = ur5.gripper.get_open_fraction()
        utils.pb_set_joint_positions(robotiq_id, [0, 2, 4, 5, 6, 7], [fract, fract, fract, -fract, fract, -fract])

        time.sleep(0.1)


main()
