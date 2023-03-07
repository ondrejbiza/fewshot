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

NPF32 = NDArray[np.float32]
NPF64 = NDArray[np.float64]


def tool0_controller_base_to_flange_base_link(T: NPF64, tf_proxy: TFProxy) -> NPF64:

    T_b_to_bl = tf_proxy.lookup_transform("base", "base_link")
    T_f_to_g = tf_proxy.lookup_transform("flange", "tool0_controller")

    return T_b_to_bl @ T @ T_f_to_g


def desk_obj_param_to_base_link_T(obj_pos: NPF64, obj_quat: NPF64, desk_center: NPF64,
                                  tf_proxy: TFProxy) -> NPF64:

    T_b_to_m = utils.pos_quat_to_transform(obj_pos + desk_center, obj_quat)
    T_bl_to_b = np.linalg.inv(tf_proxy.lookup_transform("base_link", "base"))
    return T_bl_to_b @ T_b_to_m


def to_stamped_pose_message(pos: NPF64, quat: NPF64, frame_id: str) -> geometry_msgs.msg.PoseStamped:

    msg = geometry_msgs.msg.PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose = to_pose_message(pos, quat)
    return msg


def to_pose_message(pos: NPF64, quat: NPF64) -> geometry_msgs.msg.Pose:

    msg = geometry_msgs.msg.Pose()
    # Just making sure this is actually float64. Otherwise, we get an error.
    msg.position = to_point_msg(pos.astype(np.float64))
    msg.orientation = to_quat_msg(quat.astype(np.float64))
    return msg


def to_point_msg(pos: NPF64) -> geometry_msgs.msg.Point:

    msg = geometry_msgs.msg.Point()
    msg.x = pos[0]
    msg.y = pos[1]
    msg.z = pos[2]
    return msg


def to_quat_msg(quat: NPF64) -> geometry_msgs.msg.Quaternion:

    msg = geometry_msgs.msg.Quaternion()
    msg.x = quat[0]
    msg.y = quat[1]
    msg.z = quat[2]
    msg.w = quat[3]
    return msg


def get_camera_intrinsics_and_distortion(topic: str) -> Tuple[NPF64, NPF64]:

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


def move_hand_back(pos: NPF64, quat: NPF64, delta: float) -> Tuple[NPF64, NPF64]:

    rot = Rotation.from_quat(quat).as_matrix()
    vec = np.array([0., 0., delta], dtype=np.float32)
    vec = np.matmul(rot, vec)
    pos = pos - vec
    return pos, quat


def workspace_to_base() -> NPF64:

    T = np.eye(4).astype(np.float64)
    T[:3, 3] = constants.DESK_CENTER
    return T


def tool0_controller_to_gripper_top() -> NPF64:
    """Move from the middle of the fingers towards the top of the gripper."""
    T = np.eye(4).astype(np.float64)
    T[2, 3] = -0.08  # 8 cm up
    return T


def robotiq_coupler_to_tool0() -> NPF64:
    """This is specifically for the robotiq gripper URDF."""
    rc_to_tool0_pos = np.array([0., 0., 0.004])
    rc_to_tool0_quat = Rotation.from_euler("xyz", [0., 0., -np.pi / 2]).as_quat()
    return utils.pos_quat_to_transform(rc_to_tool0_pos, rc_to_tool0_quat)


def robotiq_to_robotiq_coupler() -> NPF64:
    """This is specifically for the robotiq gripper URDF."""
    r_to_rc_pos = np.array([0., 0., 0.004])
    r_to_rc_quat = Rotation.from_euler("xyz", [0., -np.pi / 2, np.pi]).as_quat()
    return utils.pos_quat_to_transform(r_to_rc_pos, r_to_rc_quat)


def robotiq_to_tool0() -> NPF64:
    """This is specifically for the robotiq gripper URDF."""
    trans_rc_to_tool0 = robotiq_coupler_to_tool0()
    trans_r_to_rc = robotiq_to_robotiq_coupler()
    return trans_rc_to_tool0 @ trans_r_to_rc
