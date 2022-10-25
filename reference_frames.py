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

    pu.connect(use_gui=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    with pu.LockRenderer():
        with pu.HideOutput():
            floor = pu.load_model("models/short_floor.urdf")
            robot = pu.load_model(FRANKA_URDF, fixed_base=True)
            pu.assign_link_colors(robot, max_colors=3, s=0.5, v=1.)

    pu.wait_if_gui()

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
    pu.inverse_kinematics(robot, tool_link, pose)

    pu.wait_if_gui()
    pu.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument("load_path")
main(parser.parse_args())
