from dataclasses import dataclass
import functools
import time
from threading import Lock
from typing import Tuple, Optional

from numpy.typing import NDArray
import matplotlib.pyplot as plt
import rospy
import ros_numpy
from sensor_msgs.msg import Image
from skimage.transform import resize

from online_isec.tf_proxy import TFProxy


@dataclass
class ImageProxy:
    # (realsense*3)
    image_topics: Tuple[Optional[str], ...] = ("realsense_left/color/image_raw", "realsense_right/color/image_raw", "realsense_forward/color/image_raw")
    heights: Tuple[int, ...] = (720, 720, 720)
    widths: Tuple[int, ...] = (1280, 1280, 1280)

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
        with self.locks[camera_index]:
            self.images[camera_index] = image

    def get(self, camera_index: int) -> Optional[NDArray]:

        with self.locks[camera_index]:
            return self.images[camera_index]

    def close(self):

        for sub in self.image_subs:
            sub.unregister()


if __name__ == "__main__":

    # Show outputs of the proxy.
    print("Setup proxy and wait a bit.")
    rospy.init_node("image_proxy_example")
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
