from dataclasses import dataclass
import functools
from numpy.typing import NDArray
import time
from typing import Optional, Tuple, List

import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2

from src import exceptions, utils
from src.real_world import constants
from src.real_world.tf_proxy import TFProxy


@dataclass
class PointCloudProxy:

    pc_topics: Tuple[str, ...] = ("/realsense_left/depth/color/points", "/realsense_right/depth/color/points", "/realsense_forward/depth/color/points")
    nans_in_pc: Tuple[bool, ...] = (False, False, False)

    desk_center: Tuple[float, float] = (constants.DESK_CENTER[0], constants.DESK_CENTER[1])
    z_min: float = constants.DESK_CENTER[2]
    obs_clip_offset: float = 0.0
    desk_offset: float = 0.05  # TODO: what is this?

    apply_transform: bool = True

    def __post_init__(self):

        self.tf_proxy = TFProxy()

        self.msgs: List[Optional[PointCloud2]] = [None for _ in range(len(self.pc_topics))]
        self.pc_subs = []
        self.register()

    def register(self):
        """Register ros subscribers."""
        for i in range(len(self.pc_topics)):
            self.pc_subs.append(rospy.Subscriber(self.pc_topics[i], PointCloud2, functools.partial(
                self.pc_callback, camera_index=i
            ), queue_size=1))

    def unregister(self):
        """Unregister ros subscribers."""
        for sub in self.pc_subs:
            sub.unregister()

        self.pc_subs = []

    def pc_callback(self, msg: PointCloud2, camera_index: int):
        """Do not process each message to save compute."""
        self.msgs[camera_index] = msg

    def process_pc(self, msg: PointCloud2, nans_in_pc: bool):
        """Process a message on demand."""
        cloud_frame = msg.header.frame_id
        cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)

        # Comment out if you want an ordered PC.
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
            cloud[:, :3] = utils.transform_pcd(cloud[:, :3], T)

        return cloud

    def get(self, camera_index: int) -> NDArray:
        """Get a point cloud from a cameras."""
        msg = self.msgs[camera_index]
        if msg is None:
            raise exceptions.PerceptionError(f"Camera {camera_index} isn't working.")
        return self.process_pc(msg, self.nans_in_pc[camera_index])

    def get_all(self) -> NDArray:
        """Get a combined point cloud from all cameras."""
        clouds = []
        for idx, msg in enumerate(self.msgs):
            if msg is None:
                raise exceptions.PerceptionError(f"Camera {idx} isn't working.")
            clouds.append(self.process_pc(msg, self.nans_in_pc[idx]))

        return np.concatenate(clouds)

    def close(self):
        self.unregister()


@dataclass
class PointCloudProxyLeft(PointCloudProxy):
    pc_topics: Tuple[str, ...] = ("/realsense_left/depth/color/points",)
    nans_in_pc: Tuple[bool, ...] = (False,)


@dataclass
class PointCloudProxyRight(PointCloudProxy):
    pc_topics: Tuple[str, ...] = ("/realsense_right/depth/color/points",)
    nans_in_pc: Tuple[bool, ...] = (False,)


@dataclass
class PointCloudProxyForward(PointCloudProxy):
    pc_topics: Tuple[str, ...] = ("/realsense_forward/depth/color/points",)
    nans_in_pc: Tuple[bool, ...] = (False,)


if __name__ == "__main__":
    rospy.init_node("point_cloud_proxy")
    pc_proxy = PointCloudProxy()
    time.sleep(2)
    pcd = pc_proxy.get(0)
    pcd = pcd.reshape(720, 1280, 3)

    import matplotlib.pyplot as plt
    plt.subplot(1, 3, 1)
    plt.imshow(pcd[:, :, 0])
    plt.subplot(1, 3, 2)
    plt.imshow(pcd[:, :, 1])
    plt.subplot(1, 3, 3)
    plt.imshow(pcd[:, :, 2])
    plt.show()
