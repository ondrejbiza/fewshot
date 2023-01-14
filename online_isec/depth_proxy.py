import time
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
class DepthProxy:
    # (realsense, azure, structure)
    # image_topics: Tuple[Optional[str], ...] = ("cam1/color/image_raw", "/k4a/rgb_to_depth/image_raw", "/camera/ir/image")
    depth_topics: Tuple[Optional[str], ...] = ("cam1/depth/image_rect_raw", "/k4a/rgb/image_rect_color", "/camera/depth/image_rect_raw")
    heights: Tuple[int, ...] = (720, 720, 480)
    widths: Tuple[int, ...] = (1280, 1280, 640)

    def __post_init__(self):

        self.tf_proxy = TFProxy()

        self.images = [None for _ in range(len(self.depth_topics))]
        self.locks = [Lock() for _ in range(len(self.depth_topics))]

        self.image_subs = []
        for i in range(len(self.depth_topics)):
            self.image_subs.append(
                rospy.Subscriber(self.depth_topics[i], Image, functools.partial(self.image_callback, camera_index=i),
                queue_size=1))

    def image_callback(self, msg: rospy.Message, camera_index: int):

        image = ros_numpy.numpify(msg)
        with self.locks[camera_index]:
            self.images[camera_index] = image
        time.sleep(0.5)

    def get(self, camera_index: int) -> Optional[NDArray]:

        with self.locks[camera_index]:
            return self.images[camera_index]

    def close(self):

        for sub in self.image_subs:
            sub.unregister()


@dataclass
class RealsenseStructureDepthProxy(DepthProxy):
    # (realsense, structure)
    depth_topics: Tuple[Optional[str], ...] = ("cam1/aligned_depth_to_color/image_raw", "/camera/depth/image_rect_raw")
    heights: Tuple[int, ...] = (720, 480)
    widths: Tuple[int, ...] = (1280, 640)


if __name__ == "__main__":

    # Show outputs of the proxy.
    print("Setup proxy and wait a bit.")
    rospy.init_node("depth_proxy_example")
    pc_proxy = RealsenseStructureDepthProxy()
    time.sleep(2)

    camera_index = 0
    image = pc_proxy.get(camera_index)

    if image is None:
        print("Something went wrong.")
        exit(1)

    print("Image size:", image.shape)

    print("Showing depth image, scaled in meters.")
    plt.imshow((image / 1000).astype(np.float32))
    plt.show()
