from typing import Tuple
import time
import rospy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from online_isec.point_cloud_proxy import RealsenseStructurePointCloudProxy
import online_isec.utils as isec_utils
import utils


def cluster_objects_and_show(cloud: NDArray):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(utils.rotate_for_open3d(cloud))

    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))
    print("Labels:", np.unique(labels))

    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd])


def main():

    rospy.init_node("easy_perception")
    pc_proxy = RealsenseStructurePointCloudProxy()
    time.sleep(2)

    cloud = pc_proxy.get_all()
    assert cloud is not None

    cloud = cloud[..., :3]
    cloud = isec_utils.mask_workspace(cloud, (*pc_proxy.desk_center, pc_proxy.z_min))
    o3d.visualization.draw_geometries([utils.create_o3d_pointcloud(cloud)])

    cluster_objects_and_show(cloud)


main()
