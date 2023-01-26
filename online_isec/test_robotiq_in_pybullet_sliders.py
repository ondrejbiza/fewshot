import numpy as np
import pybullet as pb
import rospy
import time

from online_isec.ur5 import UR5
from pybullet_planning.pybullet_tools import utils as pu
import utils


def main():

    pu.connect(use_gui=True, show_sliders=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    robotiq = pb.loadURDF("data/robotiq.urdf", useFixedBase=True)
    pu.set_pose(robotiq, (np.array([0., 0, 0.]), np.array([1., 0., 0., 0.])))

    conf = pu.get_joint_positions(robotiq, pu.get_joints(robotiq))
    joint_names = pu.get_joint_names(robotiq, pu.get_joints(robotiq))
    joint_names = [name[11:] for name in joint_names]
    print(joint_names)

    dbg = dict()
    for i in pu.get_joints(robotiq):
        vmin, vmax = pu.get_joint_limits(robotiq, i)
        if vmin > vmax:
            tmp = vmin
            vmin = vmax
            vmax = tmp
        dbg[joint_names[i]] = pb.addUserDebugParameter(joint_names[i], vmin, vmax, conf[i])
    dbg['close'] =  pb.addUserDebugParameter('close', 1, 0, 0)

    while True:

        p = utils.read_parameters(dbg)
        conf = [p[joint_names[i]] for i in pu.get_joints(robotiq)]
        pu.set_joint_positions(robotiq, pu.get_joints(robotiq), conf)

        if p['close'] > 0:
            break

    pu.disconnect()

main()
