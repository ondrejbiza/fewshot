from typing import Tuple
import time
import rospy
import open3d as o3d
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from online_isec.point_cloud_proxy import PointCloudProxy, RealsenseStructurePointCloudProxy
import online_isec.utils as isec_utils
import utils
import viz_utils


def main():

    rospy.init_node("easy_perception")
    pc_proxy = RealsenseStructurePointCloudProxy()
    time.sleep(2)

    cloud = pc_proxy.get_all()
    assert cloud is not None

    # print("pc:")
    # o3d.visualization.draw_geometries([utils.create_o3d_pointcloud(cloud)])

    cloud = cloud[..., :3]
    cloud = isec_utils.mask_workspace(cloud, (*pc_proxy.desk_center, pc_proxy.z_min + 0.02))

    print("masked pc:")
    o3d.visualization.draw_geometries([utils.create_o3d_pointcloud(cloud)])

    mug_pc, tree_pc = isec_utils.find_mug_and_tree(cloud)

    max_size = 2000
    if len(mug_pc) > max_size:
        mug_pc, _ = utils.farthest_point_sample(mug_pc, max_size)
    if len(tree_pc) > max_size:
        tree_pc, _ = utils.farthest_point_sample(tree_pc, max_size)

    print("mug:")
    o3d.visualization.draw_geometries([utils.create_o3d_pointcloud(mug_pc)])
    print("tree:")
    o3d.visualization.draw_geometries([utils.create_o3d_pointcloud(tree_pc)])

    # load canonical objects
    with open("data/ndf_mugs_pca_8_dim.npy", "rb") as f:
        canon_mug = pickle.load(f)
    with open("data/real_tree_pc.pkl", "rb") as f:
        canon_tree_pc = pickle.load(f)["canonical_obj"]

    mug_pc_complete, _, mug_param = utils.planar_pose_warp_gd(canon_mug["pca"], canon_mug["canonical_obj"], mug_pc, object_size_reg=0.1, n_angles=12)
    tree_pc_complete, _, tree_param = utils.planar_pose_gd(canon_tree_pc, tree_pc, n_angles=12)
    viz_utils.show_scene({0: mug_pc_complete, 1: tree_pc_complete}, background=np.concatenate([mug_pc, tree_pc]))


main()
