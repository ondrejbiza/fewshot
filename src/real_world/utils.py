import time
from typing import Tuple

import geometry_msgs.msg
import numpy as np
from numpy.typing import NDArray
from sensor_msgs.msg import CameraInfo
from scipy.spatial.transform import Rotation
import rospy

from src import utils
from src.real_world import constants
from src.real_world.tf_proxy import TFProxy


def tool0_controller_base_to_flange_base_link(T: NDArray, tf_proxy: TFProxy) -> NDArray:

    T_b_to_bl = tf_proxy.lookup_transform("base", "base_link")
    T_f_to_g = tf_proxy.lookup_transform("flange", "tool0_controller")

    return T_b_to_bl @ T @ T_f_to_g


def desk_obj_param_to_base_link_T(obj_pos: NDArray, obj_quat: NDArray, desk_center: NDArray,
                                  tf_proxy: TFProxy) -> NDArray:

    T_b_to_m = utils.pos_quat_to_transform(obj_pos + desk_center, obj_quat)
    T_bl_to_b = np.linalg.inv(tf_proxy.lookup_transform("base_link", "base"))
    return T_bl_to_b @ T_b_to_m


def to_stamped_pose_message(pos: NDArray, quat: NDArray, frame_id: str) -> geometry_msgs.msg.PoseStamped:

    msg = geometry_msgs.msg.PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose = to_pose_message(pos, quat)
    return msg


def to_pose_message(pos: NDArray, quat: NDArray) -> geometry_msgs.msg.Pose:

    msg = geometry_msgs.msg.Pose()
    msg.position = to_point_msg(pos.astype(np.float64))
    msg.orientation = to_quat_msg(quat.astype(np.float64))
    return msg


def to_point_msg(pos: NDArray[np.float64]) -> geometry_msgs.msg.Point:

    msg = geometry_msgs.msg.Point()
    msg.x = pos[0]
    msg.y = pos[1]
    msg.z = pos[2]
    return msg


def to_quat_msg(quat: NDArray[np.float64]) -> geometry_msgs.msg.Quaternion:

    msg = geometry_msgs.msg.Quaternion()
    msg.x = quat[0]
    msg.y = quat[1]
    msg.z = quat[2]
    msg.w = quat[3]
    return msg


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


def move_hand_back(pos: NDArray, quat: NDArray, delta: float) -> Tuple[NDArray, NDArray]:

    rot = Rotation.from_quat(quat).as_matrix()
    vec = np.array([0., 0., delta], dtype=np.float32)
    vec = np.matmul(rot, vec)
    pos = pos - vec
    return pos, quat


def workspace_to_base():

    T = np.eye(4)
    T[:3, 3] = constants.DESK_CENTER
    return T
