import time
import os
import numpy as np
import pybullet as pb
from pybullet_planning.pybullet_tools import kuka_primitives
from pybullet_planning.pybullet_tools import utils as pu


def replace_orientation(pose, roll, pitch, yaw):

    return pu.Pose(pose[0], pu.Euler(roll, pitch, yaw))


def main():

    pu.connect(use_gui=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()
    with pu.HideOutput():
        robot = pu.load_model(pu.DRAKE_IIWA_URDF) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
        floor = pu.load_model('models/short_floor.urdf')

    mug = pu.load_model("../data/mugs/0.urdf", fixed_base=False)
    pu.set_pose(mug, pu.Pose(pu.Point(x=0.4, y=0.3, z=pu.stable_z(mug, floor)), pu.Euler(0., 0., np.pi / 4)))

    link_pose = pu.get_link_pose(mug, 0)

    link_pose = replace_orientation(link_pose, np.pi, 0., 0.)

    p = pu.Pose(point=[0, 0, 0.01])
    link_pose = pu.multiply(p, link_pose)

    pu.inverse_kinematics(robot, kuka_primitives.get_tool_link(robot), link_pose)

    pu.wait_if_gui()
    pu.disconnect()


main()
