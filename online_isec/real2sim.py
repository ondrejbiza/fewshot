import argparse
import subprocess
from typing import Tuple
import time
import rospy
import open3d as o3d
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import trimesh

from pybullet_planning.pybullet_tools import utils as pu
from online_isec.point_cloud_proxy import PointCloudProxy, RealsenseStructurePointCloudProxy
import online_isec.utils as isec_utils
import utils
import viz_utils


def main(args):

    rospy.init_node("easy_perception")
    pc_proxy = RealsenseStructurePointCloudProxy()
    time.sleep(2)

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
    with open("data/ndf_mugs_pca_8_dim.npy", "rb") as f:
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

    while True:
        pu.step_simulation()
        time.sleep(0.1)


parser = argparse.ArgumentParser()
parser.add_argument("--show-perception", default=False, action="store_true")
main(parser.parse_args())
