from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import rospy
from typing import List, Tuple, Any
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
import moveit_commander
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
    
    setup_planning: bool = False

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

        if self.setup_planning:
            name = "manipulator"
            tool_frame_id = "flange"

            self.moveit_robot = moveit_commander.RobotCommander()
            self.moveit_scene = moveit_commander.PlanningSceneInterface()
            self.moveit_move_group = moveit_commander.MoveGroupCommander(name)
            self.moveit_move_group.set_end_effector_link(tool_frame_id)

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

    def plan_and_execute_pose_target(self, base_link_to_flange_pos, base_link_to_flange_rot):

        assert self.setup_planning, "setup_planning has to be true."
        pose_msg = self.to_pose_message(base_link_to_flange_pos, base_link_to_flange_rot)
        self.moveit_move_group.set_max_velocity_scaling_factor(0.1)
        self.moveit_move_group.set_max_acceleration_scaling_factor(0.1)
        self.moveit_move_group.set_pose_target(pose_msg)

        plan_raw = self.moveit_move_group.plan()
        assert plan_raw[0], "Planning failed."

        plan = plan_raw[1]
        print(plan)
        self.moveit_move_group.execute(plan, wait=True)
        self.moveit_move_group.stop()
        self.moveit_move_group.clear_pose_targets()
        rospy.sleep(0.1)      

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

    @classmethod
    def to_pose_message(cls, pos: NDArray, quat: NDArray) -> Pose:

        msg = Pose()
        msg.position = cls.to_point_msg(pos.astype(np.float64))
        msg.orientation = cls.to_quat_msg(quat.astype(np.float64))
        return msg

    @classmethod
    def to_point_msg(cls, pos: NDArray[np.float64]) -> Point:

        msg = Point()
        msg.x = pos[0]
        msg.y = pos[1]
        msg.z = pos[2]
        return msg

    @classmethod
    def to_quat_msg(cls, quat: NDArray[np.float64]) -> Quaternion:

        msg = Quaternion()
        msg.x = quat[0]
        msg.y = quat[1]
        msg.z = quat[2]
        msg.w = quat[3]
        return msg
