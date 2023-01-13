import time
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import rospy
from sensor_msgs.msg import CameraInfo


def get_camera_intrinsics_and_distortion(topic: str) -> Tuple[NDArray, NDArray]:

    out = [False, None, None]
    def callback(msg: CameraInfo):
        out[1] = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        out[2] = np.array(msg.D, dtype=np.float64)
        out[0] = True
    
    sub = rospy.Subscriber(topic, CameraInfo, callback, queue_size=1)
    for _ in range(100):
        time.sleep(0.1)
        if out[0]:
            sub.unregister()
            return out[1], out[2]

    raise RuntimeError("Could not get camera information.")
