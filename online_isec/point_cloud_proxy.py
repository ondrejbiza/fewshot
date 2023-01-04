from typing import Tuple, Optional
from dataclasses import dataclass
from numpy.typing import NDArray
import functools

from threading import Lock
import numpy as np
import matplotlib.pyplot as plt
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import tf as not_tensorflow
import tf2_ros
import utils


@dataclass
class PointCloudProxy:
    # from the original code:
    # topics: Tuple[str] = ("/camera/depth/points", "/k4a/points2", "/cam1/depth/points")

    # Azure: pretty bad point clouds
    # topics: Tuple[str] = ("/k4a/depth_registered/points",)
    # heights: Tuple[int] = (720,)
    # widths: Tuple[int] = (1280,)
    # nans_in_pc: Tuple[bool] = (True,)

    # realsense
    topics: Tuple[str] = ("/cam1/depth/color/points",)
    heights: Tuple[int] = (480,)
    widths: Tuple[int] = (848,)
    nans_in_pc: Tuple[bool] = (False,)

    desk_center: Tuple[float, float] = (-0.527, -0.005)
    z_min: float = -0.07
    obs_clip_offset: float = 0.0
    desk_offset: float = 0.05

    def __post_init__(self):

        self.image = None
        self.cloud = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.msgs = [None for _ in range(len(self.topics))]
        # XYZRGB point cloud.
        self.clouds = [None for _ in range(len(self.topics))]
        self.images = [None for _ in range(len(self.topics))]
        # Flatten image and apply mask to turn it into a point cloud.
        # Useful if we want to have correspondence between a predicted segmentation mask
        # and the point cloud.
        self.masks = [None for _ in range(len(self.topics))]

        self.subs = [rospy.Subscriber(self.topics[i], PointCloud2, functools.partial(
            self.callback, camera_index=i, width=self.widths[i], height=self.heights[i], nans_in_pc=self.nans_in_pc[i]
        ), queue_size=1) for i in range(len(self.topics))]
        self.locks = [Lock() for _ in range(len(self.topics))]

    def callback(self, msg: rospy.Message, camera_index: int, width: int, height: int, nans_in_pc: bool):

        # Get XYZRGB point cloud from a message.
        cloud_frame = msg.header.frame_id
        pc = ros_numpy.numpify(msg)
        pc = ros_numpy.point_cloud2.split_rgb_field(pc)
        cloud = np.zeros((height * width, 6), dtype=np.float32)
        cloud[:, 0] = np.resize(pc["x"], height * width)
        cloud[:, 1] = np.resize(pc["y"], height * width)
        cloud[:, 2] = np.resize(pc["z"], height * width)
        cloud[:, 3] = np.resize(pc["r"], height * width)
        cloud[:, 4] = np.resize(pc["g"], height * width)
        cloud[:, 5] = np.resize(pc["b"], height * width)

        # We want to keep the raw image in uint8 format.
        image = cloud[:, 3: 6].reshape(height, width, 3).astype(np.uint8)

        # On the other hand, we'll keep point cloud RGB values in 0-1 floats.
        cloud[:, 3: 6] /= 255.

        if nans_in_pc:
            # Mask out NANs and keep the mask so that we can go from image to PC.
            # TODO: I added ..., 3 here, double check if there are NaNs in colors.
            mask = np.logical_not(np.isnan(cloud[..., :3]).any(axis=1))
        else:
            # If empty pixels are not NaN they should be (0, 0, 0).
            # Note the corresponding RGB values will not be NaN.
            mask = np.logical_not((cloud[..., :3] == 0).all(axis=1))

        cloud = cloud[mask]

        T = self.lookup_transform(cloud_frame, "base", rospy.Time(0))
        cloud[:, :3] = utils.transform_pointcloud_2(cloud[:, :3], T)

        with self.locks[camera_index]:
            self.msgs[camera_index] = msg
            self.clouds[camera_index] = cloud
            self.images[camera_index] = image
            self.masks[camera_index] = mask

    def lookup_transform(self, from_frame: str, to_frame: str, lookup_time=rospy.Time(0)) -> NDArray:

        transform_msg = self.tf_buffer.lookup_transform(to_frame, from_frame, lookup_time, rospy.Duration(1))
        translation = transform_msg.transform.translation
        pos = [translation.x, translation.y, translation.z]
        rotation = transform_msg.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        T = not_tensorflow.transformations.quaternion_matrix(quat)
        T[0: 3, 3] = pos
        return T

    def get(self, camera_index: int) -> Tuple[Optional[NDArray], Optional[NDArray], Optional[NDArray]]:

        with self.locks[camera_index]:
            return self.clouds[camera_index], self.images[camera_index], self.masks[camera_index]


if __name__ == "__main__":
    import time
    import open3d as o3d

    # Show outputs of the proxy.
    print("Setup proxy and wait a bit.")
    rospy.init_node("point_cloud_proxy_example")
    pc_proxy = PointCloudProxy()
    time.sleep(2)

    cloud, image, mask = pc_proxy.get(0)

    if cloud is None or image is None:
        print("Something went wrong.")
        exit(1)

    print("PC size:", cloud.shape)
    print("Image size:", image.shape)
    print("Mask size:", mask.shape)

    print("Showing RGB image.")
    plt.imshow(image)
    plt.show()

    print("Showing mask.")
    plt.imshow(mask.reshape(image.shape[0], image.shape[1]).astype(np.float32))
    plt.show()

    print("Showing point cloud.")
    colors = cloud[..., 3: 6]
    points = cloud[..., :3]
    print("Colors min,max:", colors.min(), colors.max())
    print("Points min,max:", points.min(), points.max())

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(utils.rotate_for_open3d(points))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
