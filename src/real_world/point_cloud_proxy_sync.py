from dataclasses import dataclass
import functools
from numpy.typing import NDArray
import time
from typing import Optional, Tuple
import subprocess

import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2

from src import exceptions, utils
from src.real_world import constants
from src.real_world.tf_proxy import TFProxy


@dataclass
class PointCloudProxy:
    # (realsense*3)
    pc_topics: Tuple[str, ...] = ("/realsense_left/depth/color/points", "/realsense_right/depth/color/points", "/realsense_forward/depth/color/points")
    nans_in_pc: Tuple[bool, ...] = (False, False, False)

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
            cloud[:, :3] = utils.transform_pcd(cloud[:, :3], T)

        self.clouds[camera_index] = cloud

    def get_all(self) -> Optional[NDArray]:

        self.clouds = [None for _ in range(len(self.pc_topics))]
        self.register()
        time.sleep(2)

        # subprocess.call(["rosrun", "dynamic_reconfigure", "dynparam", "set", "/realsense_left/stereo_module", "emitter_enabled", "1"])
        # subprocess.call(["rosrun", "dynamic_reconfigure", "dynparam", "set", "/realsense_right/stereo_module", "emitter_enabled", "0"])
        # subprocess.call(["rosrun", "dynamic_reconfigure", "dynparam", "set", "/realsense_forward/stereo_module", "emitter_enabled", "0"])

        # clouds = []

        # time.sleep(0.5)
        # clouds.append(self.clouds[0])

        # subprocess.call(["rosrun", "dynamic_reconfigure", "dynparam", "set", "/realsense_right/stereo_module", "emitter_enabled", "1"])
        # subprocess.call(["rosrun", "dynamic_reconfigure", "dynparam", "set", "/realsense_left/stereo_module", "emitter_enabled", "0"])

        # time.sleep(0.5)
        # clouds.append(self.clouds[1])

        # subprocess.call(["rosrun", "dynamic_reconfigure", "dynparam", "set", "/realsense_forward/stereo_module", "emitter_enabled", "1"])
        # subprocess.call(["rosrun", "dynamic_reconfigure", "dynparam", "set", "/realsense_right/stereo_module", "emitter_enabled", "0"])

        # time.sleep(0.5)
        # clouds.append(self.clouds[2])

        # subprocess.call(["rosrun", "dynamic_reconfigure", "dynparam", "set", "/realsense_forward/stereo_module", "emitter_enabled", "0"])

        clouds = []
        for cloud in self.clouds:
            clouds.append(cloud)

        for cloud in clouds:
            assert cloud is not None
        clouds = np.concatenate(clouds)

        self.unregister()

        self.clouds = [None for _ in range(len(self.pc_topics))]

        return clouds

    def close(self):
        self.unregister()


@dataclass
class PointCloudProxyLeft(PointCloudProxy):
    # (realsense*3)
    pc_topics: Tuple[str, ...] = ("/realsense_left/depth/color/points",)
    nans_in_pc: Tuple[bool, ...] = (False,)


@dataclass
class PointCloudProxyRight(PointCloudProxy):
    # (realsense*3)
    pc_topics: Tuple[str, ...] = ("/realsense_right/depth/color/points",)
    nans_in_pc: Tuple[bool, ...] = (False,)


@dataclass
class PointCloudProxyForward(PointCloudProxy):
    # (realsense*3)
    pc_topics: Tuple[str, ...] = ("/realsense_forward/depth/color/points",)
    nans_in_pc: Tuple[bool, ...] = (False,)
