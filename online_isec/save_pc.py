import argparse
from typing import Tuple
import time
import rospy
import open3d as o3d
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from online_isec.point_cloud_proxy import PointCloudProxy, RealsenseStructurePointCloudProxy
import utils
import viz_utils


def mask_workspace(cloud: NDArray, desk_center: Tuple[float, float, float], size: float=0.2) -> NDArray:

    cloud = np.copy(cloud)
    cloud[..., 0] -= desk_center[0]
    cloud[..., 1] -= desk_center[1]
    cloud[..., 2] -= desk_center[2]

    mask = np.logical_and(np.abs(cloud[..., 0]) <= size, np.abs(cloud[..., 1]) <= size)
    mask = np.logical_and(mask, cloud[..., 2] >= 0.)
    mask = np.logical_and(mask, cloud[..., 2] <= 2 * size)

    return cloud[mask]


def find_tree(cloud: NDArray) -> NDArray:

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(utils.rotate_for_open3d(cloud))

    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))

    z_max = None
    tree = None
    for label in np.unique(labels):
        pc = cloud[labels == label]
        tmp = np.max(pc[..., 2])
        if z_max is None or tmp > z_max:
            tree = pc
            z_max = tmp
    assert tree is not None, "No objects found."

    assert len(tree) > 10
    mask = tree[..., 2] >= 0.03
    tree = tree[mask]

    return tree


def update_axis(ax, new_obj: NDArray, vmin: float, vmax: float):

    ax.clear()
    ax.scatter(new_obj[:, 0], new_obj[:, 1], new_obj[:, 2], color="red")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_zlim(vmin, vmax)


def main(args):

    rospy.init_node("save_tree")
    pc_proxy = RealsenseStructurePointCloudProxy()
    time.sleep(2)

    cloud = pc_proxy.get_all()
    assert cloud is not None
    pc_proxy.close()

    # TODO: generalize to any object
    cloud = mask_workspace(cloud, (*pc_proxy.desk_center, pc_proxy.z_min + 0.02))
    o3d.visualization.draw_geometries([utils.create_o3d_pointcloud(cloud)])

    tree_pc = find_tree(cloud)
    tree_pc = tree_pc - np.mean(tree_pc, axis=0, keepdims=True)
    o3d.visualization.draw_geometries([utils.create_o3d_pointcloud(tree_pc)])

    print("# original points: {:d}".format(len(tree_pc)))
    if len(tree_pc) > args.num_points:
        tree_pc, _ = utils.farthest_point_sample(tree_pc, args.num_points)
    print("# final points: {:d}".format(len(tree_pc)))
    o3d.visualization.draw_geometries([utils.create_o3d_pointcloud(tree_pc)])

    pcd = utils.create_o3d_pointcloud(tree_pc)
    o3d.io.write_point_cloud(args.save_path, pcd)


parser = argparse.ArgumentParser()
parser.add_argument("save_path")
parser.add_argument("num_points", type=int)
main(parser.parse_args())
