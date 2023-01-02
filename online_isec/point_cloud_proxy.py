from typing import Tuple
from dataclasses import dataclass
from numpy.typing import NDArray
import functools

from threading import Lock
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import tf
import tf2_ros
import utils


@dataclass
class PointCloudProxy:
    # topics: Tuple[str] = ("/camera/depth/points", "/k4a/points2", "/cam1/depth/points")
    topics: Tuple[str] = ("/k4a/depth_registered/points",)
    desk_center: Tuple[float] = (-0.527, -0.005)
    z_min = -0.07
    obs_clip_offset: float = 0.0
    desk_offset: float = 0.05

    def __post_init__(self):

        self.image = None
        self.cloud = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.msgs = [None for _ in range(len(self.topics))]
        self.clouds = [None for _ in range(len(self.topics))]

        self.subs = [rospy.Subscriber(self.topics[i], PointCloud2, functools.partial(self.callback, camera_index=i), queue_size=1) for i in range(len(self.topics))]

    def callback(self, msg, camera_index):
        print("YY")
        cloud_frame = msg.header.frame_id
        cloud = ros_numpy.numpify(msg)
        pc = ros_numpy.numpify(msg)
        pc = ros_numpy.point_cloud2.split_rgb_field(pc)
        cloud = np.zeros((720*1280, 6), dtype=np.float32)
        cloud[:, 0] = np.resize(pc['x'], 720*1280)
        cloud[:, 1] = np.resize(pc['y'], 720*1280)
        cloud[:, 2] = np.resize(pc['z'], 720*1280)
        cloud[:, 3] = np.resize(pc['r'], 720*1280)
        cloud[:, 4] = np.resize(pc['g'], 720*1280)
        cloud[:, 5] = np.resize(pc['b'], 720*1280)
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]
        # T = self.lookup_transform(cloud_frame, "base", rospy.Time(0))
        # cloud = utils.transform_pointcloud_2(cloud, T)

        self.msgs[camera_index] = msg
        self.clouds[camera_index] = cloud

    def lookup_transform(self, from_frame, to_frame, lookup_time=rospy.Time(0)) -> NDArray:

        transform_msg = self.tf_buffer.lookup_transform(to_frame, from_frame, lookup_time, rospy.Duration(1.))
        translation = transform_msg.transform.translation
        pos = [translation.x, translation.y, translation.z]
        rotation = transform_msg.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        T = tf.transformations.quaternion_matrix(quat)
        T[0: 3, 3] = pos
        return T



rospy.init_node("test")
pc_proxy = PointCloudProxy()
import time ; time.sleep(2)
print(pc_proxy.clouds[0].shape)
print(pc_proxy.clouds[0][..., 3: 6].astype(np.uint8).min())
import open3d as o3d
pcd = o3d.geometry.PointCloud()
colors = pc_proxy.clouds[0][..., 3: 6] / 255.
print(colors.shape, colors.min(), colors.max())
pcd.points = o3d.utility.Vector3dVector(pc_proxy.clouds[0][..., :3])
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
