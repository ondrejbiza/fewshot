from typing import Tuple
import time
import rospy
import open3d as o3d
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from online_isec.point_cloud_proxy_with_images import StructureProxy
import utils
import viz_utils


def mask_workspace(cloud: NDArray, desk_center: Tuple[float, float, float], size: float=0.2) -> NDArray:

    cloud = np.copy(cloud)
    cloud[..., 0] -= desk_center[0]
    cloud[..., 1] -= desk_center[1]
    cloud[..., 2] -= desk_center[2]

    mask = np.logical_and(np.abs(cloud[..., 0]) <= size, np.abs(cloud[..., 1]) <= size)
    mask = np.logical_and(mask, cloud[..., 2] >= 0)
    mask = np.logical_and(mask, cloud[..., 2] <= 2 * size)
    return cloud[mask]


def cluster_objects(cloud: NDArray) -> Tuple[NDArray, NDArray]:

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(utils.rotate_for_open3d(cloud))

    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))
    pc1 = cloud[labels == 0]
    pc2 = cloud[labels == 1]

    assert len(pc1) > 10
    assert len(pc2) > 10

    if np.max(pc1[..., 2]) > np.max(pc2[..., 2]):
        tree = pc1
        mug = pc2
    else:
        tree = pc2
        mug = pc1
    
    return mug, tree


def main():

    rospy.init_node("easy_perception")
    pc_proxy = StructureProxy()
    time.sleep(2)

    cloud, _, _ = pc_proxy.get(0)
    assert cloud is not None

    cloud = cloud[..., :3]
    cloud = mask_workspace(cloud, (*pc_proxy.desk_center, pc_proxy.z_min))
    mug_pc, tree_pc = cluster_objects(cloud)

    print("mug:")
    o3d.visualization.draw_geometries([utils.create_o3d_pointcloud(mug_pc)])
    print("tree:")
    o3d.visualization.draw_geometries([utils.create_o3d_pointcloud(tree_pc)])

    # load canonical objects
    with open("data/ndf_mugs_pca_8_dim.npy", "rb") as f:
        canon_mug = pickle.load(f)
    with open("data/simple_trees_pca.pkl", "rb") as f:
        canon_tree = pickle.load(f)

    mug_pc_complete, _, mug_param = utils.planar_pose_warp_gd(canon_mug["pca"], canon_mug["canonical_obj"], mug_pc, object_size_reg=0.1)
    tree_pc_complete, _, tree_param = utils.planar_pose_warp_gd(canon_tree["pca"], canon_tree["canonical_obj"], tree_pc, object_size_reg=0.1)
    viz_utils.show_scene({0: mug_pc_complete, 1: tree_pc_complete}, background=np.concatenate([mug_pc, tree_pc]))


main()
