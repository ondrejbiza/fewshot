import argparse
from ast import arg
import time
import os
import pickle
import numpy as np
import open3d as o3d
import pybullet as pb

from pybullet_planning.pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_planning.pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver
from pybullet_planning.pybullet_tools import utils as pu
import utils


def replace_orientation(pose, roll, pitch, yaw):

    return pu.Pose(pose[0], pu.Euler(roll, pitch, yaw))


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
    pos, quat = pu.get_link_pose(robot, tool_link)
    print("Link position:", pos)
    print("Link orientation:", quat)

    vmin, vmax = -2, 2
    dbg = dict()
    dbg['target_x'] = pb.addUserDebugParameter('target_x', vmin, vmax, pos[0])
    dbg['target_y'] = pb.addUserDebugParameter('target_y', vmin, vmax, pos[1])
    dbg['target_z'] = pb.addUserDebugParameter('target_z', vmin, vmax, pos[2])
    dbg['close'] =  pb.addUserDebugParameter('close', 1, 0, 0)

    while True:

        p = utils.read_parameters(dbg)
        pos = pu.Point(p['target_x'], p['target_y'], p['target_z'])

        end_pose = pos, quat
        # TODO: max_distance, max_candidates?
        conf = next(either_inverse_kinematics(
            robot, info, tool_link, end_pose, max_distance=pu.INF,
            max_time=0.05, max_candidate=pu.INF, use_pybullet=False
        ), None)
        print("# Conf:", conf)

        if conf is not None:
            pu.set_joint_positions(robot, ik_joints, conf)

        if p['close'] > 0:
            break

    pu.wait_if_gui()
    pu.disconnect()


parser = argparse.ArgumentParser()
main(parser.parse_args())
