from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import rospy
from typing import List, Tuple, Any
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from online_isec.robotiq_gripper import Gripper
from online_isec.tf_proxy import TFProxy
from online_isec import transformation


@dataclass
class UR5:

    pick_offset: float = 0.1
    place_offset: float = 0.1
    place_open_pos: float = 0.

    home_joint_values: Tuple[float, ...] = (0.65601951, -1.76965791, 1.79728603, -1.60219127, -1.5338834, 2.21791005)
    desk_joint_values: Tuple[float, ...] = (-0.1835673491, -1.446624104, 1.77286005, -1.90146142, -1.532696072, 1.339956641)
    joint_names_speedj: Tuple[str, ...] = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')
    
    def __post_init__(self):

        self.gripper = Gripper(True)
        self.gripper.reset()
        self.gripper.activate()

        # current joint values
        self.joint_values = np.array([0] * 6)
        self.joints_sub = rospy.Subscriber("/joint_states", JointState, self.joints_callback)

        # commanding the arm
        self.pub = rospy.Publisher("/ur_hardware_interface/script_command", String, queue_size=10)

        self.tf_proxy = TFProxy()

    def move_to_j(self, joint_pos: Tuple[float, float, float, float, float, float], speed: float=0.5):

        self.pub.publish("stopl(2)")
        rospy.sleep(0.2)
        for _ in range(5):
            s = "movej({}, v={:f})".format(list(joint_pos), speed)
            self.pub.publish(s)
            self.wait_until_not_moving()
            if np.allclose(joint_pos, self.joint_values, atol=1e-2):
                break

    def wait_until_not_moving(self):

        while True:
            prev_joint_position = self.joint_values.copy()
            rospy.sleep(0.2)
            if np.allclose(prev_joint_position, self.joint_values, atol=1e-3):
                break

    def joints_callback(self, msg: JointState):

        positions_dict = {}
        for i in range(len(msg.position)):
            positions_dict[msg.name[i]] = msg.position[i]
        self.joint_values = np.array([positions_dict[i] for i in self.joint_names_speedj])

    def get_end_effector_pose(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

        T = self.tf_proxy.lookup_transform("tool0_controller", "base")
        pos = T[:3, 3]
        rot_q = transformation.quaternion_from_matrix(T)
        return pos, rot_q
