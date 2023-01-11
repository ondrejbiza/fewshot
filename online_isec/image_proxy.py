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
class ImageProxy:
    # (realsense, azure, structure)
    image_topics: Tuple[Optional[str], ...] = ("cam1/color/image_raw", "/k4a/rgb/image_rect_color", "/camera/ir/image")
    heights: Tuple[int, ...] = (720, 720, 480)
    widths: Tuple[int, ...] = (1280, 1280, 640)

    def __post_init__(self):

        self.tf_proxy = TFProxy()

        self.images = [None for _ in range(len(self.image_topics))]
        self.locks = [Lock() for _ in range(len(self.image_topics))]

        self.image_subs = []
        for i in range(len(self.image_topics)):
            self.image_subs.append(
                rospy.Subscriber(self.image_topics[i], Image, functools.partial(self.image_callback, camera_index=i),
                queue_size=1))

    def image_callback(self, msg: rospy.Message, camera_index: int):

        image = ros_numpy.numpify(msg)
        time.sleep(0.5)
        with self.locks[camera_index]:
            self.images[camera_index] = image

    def get(self, camera_index: int) -> Optional[NDArray]:

        with self.locks[camera_index]:
            return self.images[camera_index]


if __name__ == "__main__":

    # Show outputs of the proxy.
    print("Setup proxy and wait a bit.")
    rospy.init_node("point_cloud_proxy_example")
    pc_proxy = ImageProxy()
    time.sleep(2)

    camera_index = 2
    image = pc_proxy.get(camera_index)

    if image is None:
        print("Something went wrong.")
        exit(1)

    print("Image size:", image.shape)

    print("Showing RGB image.")
    plt.imshow(image)
    plt.show()
