from typing import Tuple, Optional
from dataclasses import dataclass
from numpy.typing import NDArray
import functools

from threading import Lock
import numpy as np
import matplotlib.pyplot as plt
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
from skimage.transform import resize
import tf as not_tensorflow
import tf2_ros
import utils

from online_isec.tf_proxy import TFProxy


@dataclass
class PointCloudProxy:
    # (realsense, azure, structure)
    pc_topics: Tuple[str, ...] = ("/cam1/depth/color/points", "/k4a/depth_registered/points", "/camera/depth/points")
    nans_in_pc: Tuple[bool, ...] = (False, True, True)

    desk_center: Tuple[float, float] = (-0.527, -0.005)
    z_min: float = -0.07
    obs_clip_offset: float = 0.0
    desk_offset: float = 0.05  # TODO: what is this?

    def __post_init__(self):

        self.image = None
        self.cloud = None

        self.tf_proxy = TFProxy()

        self.msgs = [None for _ in range(len(self.pc_topics))]
        # XYZRGB point cloud.
        self.clouds = [None for _ in range(len(self.pc_topics))]
        self.locks = [Lock() for _ in range(len(self.pc_topics))]

        self.pc_subs = []
        for i in range(len(self.pc_topics)):
            # Get point clouds.
            self.pc_subs.append(rospy.Subscriber(self.pc_topics[i], PointCloud2, functools.partial(
                self.pc_callback, camera_index=i, nans_in_pc=self.nans_in_pc[i]
            ), queue_size=1))

    def pc_callback(self, msg: rospy.Message, camera_index: int, nans_in_pc: bool):

        # Get XYZRGB point cloud from a message.
        cloud_frame = msg.header.frame_id

        cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)

        if nans_in_pc:
            # Mask out NANs and keep the mask so that we can go from image to PC.
            # TODO: I added ..., 3 here, double check if there are NaNs in colors.
            mask = np.logical_not(np.isnan(cloud[..., :3]).any(axis=1))
        else:
            # If empty pixels are not NaN they should be (0, 0, 0).
            # Note the corresponding RGB values will not be NaN.
            mask = np.logical_not((cloud[..., :3] == 0).all(axis=1))

        cloud = cloud[mask]

        T = self.tf_proxy.lookup_transform(cloud_frame, "base", rospy.Time(0))
        cloud[:, :3] = utils.transform_pointcloud_2(cloud[:, :3], T)

        time.sleep(1.)
        with self.locks[camera_index]:
            self.msgs[camera_index] = msg
            self.clouds[camera_index] = cloud

    def get(self, camera_index: int) -> Optional[NDArray]:

        with self.locks[camera_index]:
            return self.clouds[camera_index]


if __name__ == "__main__":
    import time
    import open3d as o3d

    # Show outputs of the proxy.
    print("Setup proxy and wait a bit.")
    rospy.init_node("point_cloud_proxy_example")
    pc_proxy = PointCloudProxy()
    time.sleep(2000000)

    camera_index = 1
    cloud = pc_proxy.get(camera_index)

    if cloud is None:
        print("Something went wrong.")
        exit(1)

    print("PC size:", cloud.shape)

    print("Showing point cloud.")
    print("Points min,max:", cloud.min(), cloud.max())

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(utils.rotate_for_open3d(cloud))
    o3d.visualization.draw_geometries([pcd])
