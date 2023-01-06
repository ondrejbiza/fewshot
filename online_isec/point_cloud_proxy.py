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


@dataclass
class PointCloudProxy:
    # (realsense, azure, structure)
    pc_topics: Tuple[str, ...] = ("/cam1/depth/color/points", "/k4a/depth_registered/points", "/camera/depth/points")
    image_topics: Tuple[Optional[str], ...] = ("cam1/color/image_raw", None, None)
    save_image: Tuple[bool, ...] = (True, True, False)
    heights: Tuple[int, ...] = (720, 720, 480)
    widths: Tuple[int, ...] = (1280, 1280, 640)
    nans_in_pc: Tuple[bool, ...] = (False, True, True)

    desk_center: Tuple[float, float] = (-0.527, -0.005)
    z_min: float = -0.07
    obs_clip_offset: float = 0.0
    desk_offset: float = 0.05

    def __post_init__(self):

        self.image = None
        self.cloud = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.msgs = [None for _ in range(len(self.pc_topics))]
        # XYZRGB point cloud.
        self.clouds = [None for _ in range(len(self.pc_topics))]
        self.images = [None for _ in range(len(self.pc_topics))]
        # Flatten image and apply mask to turn it into a point cloud.
        # Useful if we want to have correspondence between a predicted segmentation mask
        # and the point cloud.
        self.masks = [None for _ in range(len(self.pc_topics))]
        self.locks = [Lock() for _ in range(len(self.pc_topics))]

        self.pc_subs = []
        self.image_subs = []
        for i in range(len(self.pc_topics)):
            save_image_from_pc = (self.image_topics[i] is None) and self.save_image[i]
            # Get point clouds.
            self.pc_subs.append(rospy.Subscriber(self.pc_topics[i], PointCloud2, functools.partial(
                self.pc_callback, camera_index=i, width=self.widths[i], height=self.heights[i], nans_in_pc=self.nans_in_pc[i],
                save_image=save_image_from_pc
            ), queue_size=1))

            # Some point clouds return blank RGB values for depth pixels that do not have a valid value.
            # This means we will get an ugly image from the XYZRGB point cloud.
            # Hence, we need to have one subscriber for the point cloud and one for the image.
            # It is important to make sure the image and the point cloud are aligned (i.e. image.reshape(-1, 3)[mask] ~= cloud[..., 3: 6]).
            if self.image_topics[i] is not None:
                self.image_subs.append(
                    rospy.Subscriber(self.image_topics[i], Image, functools.partial(self.image_callback, camera_index=i),
                    queue_size=1))
            else:
                self.image_subs.append(None)

    def pc_callback(self, msg: rospy.Message, camera_index: int, width: int, height: int, nans_in_pc: bool, save_image: bool):

        # Get XYZRGB point cloud from a message.
        cloud_frame = msg.header.frame_id
        pc = ros_numpy.numpify(msg)
        if save_image:
            pc = ros_numpy.point_cloud2.split_rgb_field(pc)

        # Get point cloud.
        cloud = np.zeros((height * width, 6), dtype=np.float32)
        cloud[:, 0] = np.resize(pc["x"], height * width)
        cloud[:, 1] = np.resize(pc["y"], height * width)
        cloud[:, 2] = np.resize(pc["z"], height * width)

        if save_image:
            # Get point colors.
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
            if save_image:
                self.images[camera_index] = image
            self.masks[camera_index] = mask

    def image_callback(self, msg: rospy.Message, camera_index: int):

        image = ros_numpy.numpify(msg)
        with self.locks[camera_index]:
            self.images[camera_index] = image

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


@dataclass
class RealsenseAzureProxy:
    # (realsense, azure)
    pc_topics: Tuple[str, ...] = ("/cam1/depth/color/points", "/k4a/depth_registered/points")
    image_topics: Tuple[Optional[str], ...] = ("cam1/color/image_raw", None)
    save_image: Tuple[bool, ...] = (True, True)
    heights: Tuple[int, ...] = (720, 720)
    widths: Tuple[int, ...] = (1280, 1280)
    nans_in_pc: Tuple[bool, ...] = (False, True)


@dataclass
class StructureProxy(PointCloudProxy):
    pc_topics: Tuple[str, ...] = ("/camera/depth/points",)
    image_topics: Tuple[Optional[str], ...] = (None,)
    save_image: Tuple[bool, ...] = (False,)
    heights: Tuple[int, ...] = (480,)
    widths: Tuple[int, ...] = (640,)
    nans_in_pc: Tuple[bool, ...] = (True,)


if __name__ == "__main__":
    import time
    import open3d as o3d

    # Show outputs of the proxy.
    print("Setup proxy and wait a bit.")
    rospy.init_node("point_cloud_proxy_example")
    pc_proxy = PointCloudProxy()
    time.sleep(2)

    camera_index = 1
    cloud, image, mask = pc_proxy.get(camera_index)

    if cloud is None:
        print("Something went wrong.")
        exit(1)

    print("PC size:", cloud.shape)
    if image is not None:
        print("Image size:", image.shape)
    print("Mask size:", mask.shape)

    if image is not None:
        print("Showing RGB image.")
        plt.imshow(image)
        plt.show()

    print("Showing mask.")
    plt.imshow(mask.reshape(pc_proxy.heights[camera_index], pc_proxy.widths[camera_index]).astype(np.float32))
    plt.show()

    print("Showing point cloud.")
    colors = cloud[..., 3: 6]
    points = cloud[..., :3]
    print("Colors min,max:", colors.min(), colors.max())
    print("Points min,max:", points.min(), points.max())

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(utils.rotate_for_open3d(points))
    if image is not None:
        # If image is None, force open3d to use default colors.
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
