import argparse
from ast import arg
import time
import os
import pickle
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, List, Any, Optional
import open3d as o3d
import pybullet as pb

from pybullet_planning.pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_planning.pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver
from pybullet_planning.pybullet_tools import utils as pu
import utils


def move_hand_back(pose: Tuple[NDArray, NDArray], delta: float) -> Tuple[NDArray, NDArray]:

    pos, quat = pose
    rot = pu.matrix_from_quat(quat)
    vec = np.array([0., 0., delta], dtype=np.float32)
    vec = np.matmul(rot, vec)
    pos -= vec
    return pos, quat


def main(args):

    pu.connect(use_gui=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    with pu.LockRenderer():
        with pu.HideOutput():
            floor = pu.load_model("models/short_floor.urdf")
            robot = pu.load_model(FRANKA_URDF, fixed_base=True)
            pu.assign_link_colors(robot, max_colors=3, s=0.5, v=1.)

    info = PANDA_INFO
    print(info)
    tool_link = pu.link_from_name(robot, "panda_hand")
    ik_joints = get_ik_joints(robot, info, tool_link)

    pose = pu.get_link_pose(robot, tool_link)
    initial_pos = pu.Point(0., 0.27, 0.59)
    pose = (initial_pos, pose[1])

    conf = next(either_inverse_kinematics(
        robot, info, tool_link, pose, max_distance=pu.INF,
        max_time=pu.INF, max_attempts=200, use_pybullet=False
    ), None)
    assert conf is not None
    pu.set_joint_positions(robot, ik_joints, conf)

    mug = pu.load_model("../data/mugs/test/0.urdf", fixed_base=False)
    pu.set_pose(mug, pu.Pose(pu.Point(x=0.4, y=0.3, z=pu.stable_z(mug, floor)), pu.Euler(0., 0., np.pi / 4)))

    # get registered points on the warped canonical object
    indices = np.load(args.load_path)
    print("Registered index:", indices)

    # get a point cloud
    pcs, _ = utils.observe_point_cloud(utils.RealSenseD415.CONFIG, [2])
    # pcd = utils.create_o3d_pointcloud(pcs[2])
    # utils.o3d_visualize(pcd)

    # load canonical objects
    canon = {}
    with open("data/mugs_pca.pkl", "rb") as f:
        canon[2] = pickle.load(f)

    # fit canonical objects to observed point clouds
    new_obj_1, _, _ = utils.planar_pose_warp_gd(canon[2]["pca"], canon[2]["canonical_obj"], pcs[2])
    # pcd = utils.create_o3d_pointcloud(new_obj_1)
    # colors = np.zeros_like(new_obj_1)
    # colors[indices, 0] = 1.
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # utils.o3d_visualize(pcd)

    position = new_obj_1[indices]
    print("Position:", position)

    tool_link = pu.link_from_name(robot, "panda_hand")
    pose = pu.Pose(position, pu.Euler(roll=np.pi))
    pose = move_hand_back(pose, 0.15)

    saved_world = pu.WorldSaver()

    conf = next(either_inverse_kinematics(
        robot, info, tool_link, pose, max_distance=pu.INF,
        max_time=pu.INF, max_attempts=200, use_pybullet=False
    ), None)
    assert conf is not None
    pu.set_joint_positions(robot, ik_joints, conf)

    # conf = pu.inverse_kinematics(robot, tool_link, pose)
    # assert conf is not None

    # TODO: do we have to restore the world here?
    saved_world.restore()

    # TODO: do ik_joints and conf align?
    path = pu.plan_joint_motion(robot, ik_joints, conf, obstacles=[mug])
    assert path is not None

    saved_world.restore()

    pu.wait_if_gui()

    # TODO: there's some kind of a real-time joint controller
    # pu.joint_controller, pu.enable_real_time, pu.enable_gravity
    for conf in path:
        pu.set_joint_positions(robot, ik_joints, conf)
        pu.wait_for_duration(0.005)

    pu.wait_if_gui()

    pu.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument("load_path")
main(parser.parse_args())
