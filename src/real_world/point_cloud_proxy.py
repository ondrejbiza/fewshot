from dataclasses import dataclass
import functools
from numpy.typing import NDArray
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
