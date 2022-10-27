import argparse
import os
import numpy as np
import open3d as o3d
import pybullet as pb

from pybullet_planning.pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_planning.pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver
from pybullet_planning.pybullet_tools import utils as pu
import utils


def main(args):

    pu.connect(use_gui=True, show_sliders=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    with pu.LockRenderer():
        with pu.HideOutput():
            floor = pu.load_model("models/short_floor.urdf")
            # robot = pu.load_model(FRANKA_URDF, fixed_base=True)
            robot = pu.load_pybullet(os.path.join("pybullet_planning", FRANKA_URDF), fixed_base=True)
            pu.assign_link_colors(robot, max_colors=3, s=0.5, v=1.)

    pu.dump_body(robot)

    info = PANDA_INFO
    tool_link = pu.link_from_name(robot, "panda_hand")
    ik_joints = get_ik_joints(robot, info, tool_link)

    f1 = pu.joint_from_name(robot, "panda_finger_joint1")
    f2 = pu.joint_from_name(robot, "panda_finger_joint2")
    ik_joints = ik_joints + [f1, f2]

    conf = pu.get_joint_positions(robot, ik_joints)
    pos, quat = pu.get_link_pose(robot, tool_link)
    print("Link position:", pos)
    print("Link orientation:", quat)

    dbg = dict()
    for i in range(len(ik_joints)):
        vmin, vmax = pu.get_joint_limits(robot, ik_joints[i])
        dbg['joint_{:d}'.format(i)] = pb.addUserDebugParameter('joint_{:d}'.format(i), vmin, vmax, conf[i])
    dbg['close'] =  pb.addUserDebugParameter('close', 1, 0, 0)

    while True:

        p = utils.read_parameters(dbg)
        conf = [p['joint_{:d}'.format(i)] for i in range(len(ik_joints))]
        pu.set_joint_positions(robot, ik_joints, conf)

        if p['close'] > 0:
            break

    pu.disconnect()


parser = argparse.ArgumentParser()
main(parser.parse_args())
