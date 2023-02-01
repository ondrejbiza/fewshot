from typing import Tuple, Optional
from dataclasses import dataclass
from numpy.typing import NDArray
import functools
import time

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
from online_isec import constants
import exceptions


@dataclass
class PointCloudProxy:
    # (realsense, azure, structure)
    pc_topics: Tuple[str, ...] = ("/cam1/depth/color/points", "/k4a/depth_registered/points", "/camera/depth/points")
    nans_in_pc: Tuple[bool, ...] = (False, True, True)

    desk_center: Tuple[float, float] = (constants.DESK_CENTER[0], constants.DESK_CENTER[1])
    z_min: float = constants.DESK_CENTER[2]
    obs_clip_offset: float = 0.0
    desk_offset: float = 0.05  # TODO: what is this?

    apply_transform: bool = True

    def __post_init__(self):

        self.tf_proxy = TFProxy()

        self.clouds = [None for _ in range(len(self.pc_topics))]
        self.pc_subs = []

    def register(self):

        for i in range(len(self.pc_topics)):
            # Get point clouds.
            self.pc_subs.append(rospy.Subscriber(self.pc_topics[i], PointCloud2, functools.partial(
                self.pc_callback, camera_index=i, nans_in_pc=self.nans_in_pc[i]
            ), queue_size=1))

        time.sleep(0.5)

    def unregister(self):

        for sub in self.pc_subs:
            sub.unregister()

        # No idea if a callback is still running while I unregister ...
        time.sleep(0.5)

        self.pc_subs = []

    def pc_callback(self, msg: PointCloud2, camera_index: int, nans_in_pc: bool):

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

        if self.apply_transform:
            T = self.tf_proxy.lookup_transform(cloud_frame, "base", rospy.Time(0))
            cloud[:, :3] = utils.transform_pointcloud_2(cloud[:, :3], T)

        self.clouds[camera_index] = cloud

    def get_all(self) -> Optional[NDArray]:

        self.clouds = [None for _ in range(len(self.pc_topics))]
        self.register()

        for _ in range(100):
            time.sleep(0.1)
            flag = True
            for cloud in self.clouds:
                if cloud is None:
                    flag = False
            if flag:
                break

        self.unregister()

        clouds = []
        for cloud in self.clouds:
            if cloud is None:
                raise exceptions.PerceptionError("A subset of depths sensors doesn't work.")
            clouds.append(cloud)
        clouds = np.concatenate(clouds)

        self.clouds = [None for _ in range(len(self.pc_topics))]

        return clouds

    def close(self):
        self.unregister()


@dataclass
class RealsenseStructurePointCloudProxy(PointCloudProxy):
    # (realsense, structure)
    pc_topics: Tuple[str, ...] = ("/cam1/depth/color/points", "/camera/depth/points")
    nans_in_pc: Tuple[bool, ...] = (False, True)


if __name__ == "__main__":
    import time
    import open3d as o3d

    # Show outputs of the proxy.
    print("Setup proxy and wait a bit.")
    rospy.init_node("point_cloud_proxy_example")
    pc_proxy = PointCloudProxy()

    cloud = pc_proxy.get_all()

    if cloud is None:
        print("Something went wrong.")
        exit(1)

    print("PC size:", cloud.shape)

    print("Showing point cloud.")
    print("Points min,max:", cloud.min(), cloud.max())

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(utils.rotate_for_open3d(cloud))
    o3d.visualization.draw_geometries([pcd])
