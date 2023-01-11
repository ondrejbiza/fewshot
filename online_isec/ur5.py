from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import rospy
from typing import List, Tuple, Any
from sensor_msgs.msg import JointState
from online_isec.robotiq_gripper import Gripper
from online_isec.tf_proxy import TFProxy
from online_isec import transformation


@dataclass
class UR5:

    pick_offset: float = 0.1
    place_offset: float = 0.1
    place_open_pos: float = 0.

    home_0_joint_values: Tuple[float, ...] = (-0.59910423, -1.46224243,  1.50362396, -1.61612159, -1.53385908, 0.96311188)
    home_1_joint_values: Tuple[float, ...] = ( 0.13459259, -1.48815376,  1.53233957, -1.61879331, -1.5339554 , 1.69678915)
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

        self.tf_proxy = TFProxy()

    def joints_callback(self, msg: rospy.Message):

        positions_dict = {}
        for i in range(len(msg.position)):
            positions_dict[msg.name[i]] = msg.position[i]
        self.joint_values = np.array([positions_dict[i] for i in self.joint_names_speedj])

    def get_end_effector_pose(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

        T = self.tf_proxy.lookup_transform("base", "tool0_controller")
        pos = T[:3, 3]
        rot_q = transformation.quaternion_from_matrix(T)
        return pos, rot_q
