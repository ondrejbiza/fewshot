from typing import Tuple
import time
import rospy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from online_isec.point_cloud_proxy import StructureProxy
import utils


def mask_workspace(cloud: NDArray, desk_center: Tuple[float, float, float], size: float=0.2) -> NDArray:

    cloud = np.copy(cloud)
    cloud[..., 0] -= desk_center[0]
    cloud[..., 1] -= desk_center[1]
    cloud[..., 2] -= desk_center[2]

    mask = np.logical_and(np.abs(cloud[..., 0]) <= size, np.abs(cloud[..., 1]) <= size)
    mask = np.logical_and(mask, cloud[..., 2] >= 0)
    mask = np.logical_and(mask, cloud[..., 2] <= 2 * size)
    return cloud[mask]


def cluster_objects_and_show(cloud: NDArray):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(utils.rotate_for_open3d(cloud))

    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))

    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd])


def main():

    rospy.init_node("easy_perception")
    pc_proxy = StructureProxy()
    time.sleep(2)

    cloud, _, _ = pc_proxy.get(0)
    assert cloud is not None

    cloud = cloud[..., :3]
    cloud = mask_workspace(cloud, (*pc_proxy.desk_center, pc_proxy.z_min))
    cluster_objects_and_show(cloud)


main()
