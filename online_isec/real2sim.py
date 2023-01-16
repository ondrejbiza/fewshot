import argparse
import subprocess
import threading
from typing import Tuple
import time
import rospy
import open3d as o3d
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import trimesh
import pybullet as pb
from scipy.spatial.transform import Rotation

from pybullet_planning.pybullet_tools import utils as pu
from online_isec.point_cloud_proxy import PointCloudProxy, RealsenseStructurePointCloudProxy
import online_isec.utils as isec_utils
from online_isec.ur5 import UR5
from online_isec import constants
import utils
import viz_utils


def worker(ur5, sphere, mug, data):

    while True:

        if data["stop"]:
            break

        pos, quat = ur5.get_end_effector_pose()

        if data["T"] is not None:
            T_g = utils.pos_quat_to_transform(pos - constants.DESK_CENTER, quat)
            T = np.matmul(T_g, data["T"])

            m_pos, m_quat = utils.transform_to_pos_quat(T)
            pu.set_pose(mug, (m_pos, m_quat))

        pu.set_position(sphere, pos[0] - constants.DESK_CENTER[0], pos[1] - constants.DESK_CENTER[1], pos[2] - constants.DESK_CENTER[2])
        time.sleep(0.1)


def save_pick_pose(filled_and_transformed_mug, gripper_pos, gripper_rot):

    dist = np.sqrt(np.sum(np.square(filled_and_transformed_mug - gripper_pos), axis=1))
    index = np.argmin(dist)
    with open("data/real_pick_clone.pkl", "wb") as f:
        pickle.dump({
            "index": index,
            "quat": gripper_rot
        }, f)


def save_place_contact_points(mug, tree):

    spheres = []

    pb.performCollisionDetection()

    # cols = pb.getContactPoints(mug, tree)
    cols = pb.getClosestPoints(mug, tree, 0.01)

    for s in spheres:
        pu.remove_body(s)
    spheres = []

    for col in cols:
        pos = col[6]
        with pu.HideOutput():
            s = pu.load_model("../data/sphere.urdf")
            pu.set_pose(s, pu.Pose(pu.Point(*pos)))
        spheres.append(s)

    pos_1 = [col[5] for col in cols]
    pos_2 = [col[6] for col in cols]

    pos_1 = np.stack(pos_1, axis=0).astype(np.float32)
    pos_2 = np.stack(pos_2, axis=0).astype(np.float32)

    # TODO: Perform the rest of place cloning.
    input("Done?")


def main(args):

    rospy.init_node("easy_perception")
    pc_proxy = RealsenseStructurePointCloudProxy()
    ur5 = UR5()
    time.sleep(2)
    ur5.move_to_j(ur5.home_joint_values)

    cloud = pc_proxy.get_all()
    assert cloud is not None

    cloud = cloud[..., :3]
    cloud = isec_utils.mask_workspace(cloud, (*pc_proxy.desk_center, pc_proxy.z_min + 0.02))

    if args.show_perception:
        print("masked pc:")
        o3d.visualization.draw_geometries([utils.create_o3d_pointcloud(cloud)])

    mug_pc, tree_pc = isec_utils.find_mug_and_tree(cloud)

    max_size = 2000
    if len(mug_pc) > max_size:
        mug_pc, _ = utils.farthest_point_sample(mug_pc, max_size)
    if len(tree_pc) > max_size:
        tree_pc, _ = utils.farthest_point_sample(tree_pc, max_size)

    if args.show_perception:
        print("mug:")
        o3d.visualization.draw_geometries([utils.create_o3d_pointcloud(mug_pc)])
        print("tree:")
        o3d.visualization.draw_geometries([utils.create_o3d_pointcloud(tree_pc)])

    # load canonical objects
    with open("data/ndf_mugs_pca_4_dim.npy", "rb") as f:
        canon_mug = pickle.load(f)
    with open("data/real_tree_pc.pkl", "rb") as f:
        canon_tree = pickle.load(f)

    mug_pc_complete, _, mug_param = utils.planar_pose_warp_gd(canon_mug["pca"], canon_mug["canonical_obj"], mug_pc, object_size_reg=0.1, n_angles=12)
    tree_pc_complete, _, tree_param = utils.planar_pose_gd(canon_tree["canonical_obj"], tree_pc, n_angles=12)

    if args.show_perception:
        viz_utils.show_scene({0: mug_pc_complete, 1: tree_pc_complete}, background=np.concatenate([mug_pc, tree_pc]))

    vertices = (canon_mug["canonical_obj"] + canon_mug["pca"].inverse_transform(mug_param[0]).reshape((-1, 3)))[:len(canon_mug["canonical_mesh_points"])]
    mesh = trimesh.base.Trimesh(vertices=vertices, faces=canon_mug["canonical_mesh_faces"])
    mesh.export("tmp.stl")
    # subprocess.call(["admesh", "--write-binary-stl={:s}".format("tmp.stl"), "tmp.stl"])

    # TODO: Resolution=1M was taking too long, but resolution=100k is not very good.
    convex_meshes = trimesh.decomposition.convex_decomposition(
        mesh, resolution=1000000, depth=20, concavity=0.0025, planeDownsampling=4, convexhullDownsampling=4,
        alpha=0.05, beta=0.05, gamma=0.00125, pca=0, mode=0, maxNumVerticesPerCH=256, minVolumePerCH=0.0001,
        convexhullApproximation=1, oclDeviceID=0
    )

    decomposed_scene = trimesh.scene.Scene()
    for i, convex_mesh in enumerate(convex_meshes):
        decomposed_scene.add_geometry(convex_mesh, node_name="hull_{:d}".format(i))
    decomposed_scene.export("tmp.obj", file_type="obj")

    pu.connect(use_gui=True, show_sliders=False)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    mug = pu.load_model("../tmp.urdf", pu.Pose(mug_param[1], pu.Euler(yaw=mug_param[2])))
    tree = pu.load_model("../data/real_tree.urdf", pu.Pose(tree_param[0], pu.Euler(yaw=tree_param[1])), fixed_base=True)

    point = pu.load_model("../data/sphere.urdf")
    data = {
        "T": None,
        "stop": False,
    }

    thread = threading.Thread(target=worker, args=(ur5, point, mug, data))
    thread.start()
    input("Close gripper?")

    gripper_pos, gripper_rot = ur5.get_end_effector_pose()
    gripper_pos = gripper_pos - constants.DESK_CENTER

    # TODO: We do not account for the mug moving as it is picked.
    save_pick_pose(mug_pc_complete, gripper_pos, gripper_rot)

    T_g = utils.pos_quat_to_transform(gripper_pos, gripper_rot)

    mug_pos, mug_rot = pu.get_pose(mug)
    T_m = utils.pos_quat_to_transform(mug_pos, mug_rot)

    T = np.matmul(np.linalg.inv(T_g), T_m)
    data["T"] = T

    ur5.gripper.close_gripper()

    input("Open gripper?")

    data["stop"] = True
    thread.join()

    save_place_contact_points(mug, tree)

    ur5.gripper.open_gripper()
    ur5.move_to_j(ur5.home_joint_values)


parser = argparse.ArgumentParser()
parser.add_argument("--show-perception", default=False, action="store_true")
main(parser.parse_args())
