from dataclasses import dataclass
import time
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
import rospy
from typing import List, Tuple, Any
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.msg import DisplayTrajectory
import moveit_commander
from online_isec.robotiq_gripper import Gripper
from online_isec.tf_proxy import TFProxy
from online_isec import transformation
import exceptions
from online_isec import constants
import utils
import online_isec.utils as isec_utils
from online_isec.rviz_pub import RVizPub


@dataclass
class UR5:

    pick_offset: float = 0.1
    place_offset: float = 0.1
    place_open_pos: float = 0.

    home_joint_values: Tuple[float, ...] = (0.65601951, -1.76965791, 1.79728603, -1.60219127, -1.5338834, -0.785)
    desk_joint_values: Tuple[float, ...] = (-0.1835673491, -1.446624104, 1.77286005, -1.90146142, -1.532696072, 1.339956641)
    joint_names_speedj: Tuple[str, ...] = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')
    
    setup_planning: bool = False
    move_group_name: str = "manipulator"
    tool_frame_id: str = "flange"

    def __post_init__(self):

        self.tf_proxy = TFProxy()
        self.rviz_pub = RVizPub()

        self.gripper = Gripper(True)
        self.gripper.reset()
        self.gripper.activate()

        # current joint values
        self.joint_values = np.array([0] * 6)
        self.joints_sub = rospy.Subscriber("/joint_states", JointState, self.joints_callback)

        if not self.setup_planning:
            # commanding the arm using ur script
            self.pub = rospy.Publisher("/ur_hardware_interface/script_command", String, queue_size=10)


        if self.setup_planning:
            self.moveit_robot = moveit_commander.RobotCommander()
            self.moveit_scene = moveit_commander.PlanningSceneInterface()
            self.moveit_move_group = moveit_commander.MoveGroupCommander(self.move_group_name)
            self.moveit_move_group.set_end_effector_link(self.tool_frame_id)
            rospy.sleep(2)

            self.moveit_scene.clear()
            assert isec_utils.check_clean_moveit_scene(self.moveit_scene)

            desk_center = utils.transform_pointcloud_2(np.array(constants.DESK_CENTER)[None], self.tf_proxy.lookup_transform("base", "base_link"))[0]
            pose = isec_utils.to_stamped_pose_message(desk_center, np.array([1., 0., 0., 0.]), "base_link")
            self.moveit_scene.add_box("table", pose, [2., 2., 0.001])
            assert isec_utils.check_added_to_moveit_scene("table", self.moveit_scene)

            pose = isec_utils.to_stamped_pose_message(np.array([0., 0., -0.04]), np.array([1., 0., 0., 0.]), "base_link")
            self.moveit_scene.add_box("robot_box", pose, [0.16, 0.5, 0.075])
            assert isec_utils.check_added_to_moveit_scene("robot_box", self.moveit_scene)

            tmp = np.copy(desk_center)
            tmp[1] -= 0.17 + 0.2 + 0.1
            pose = isec_utils.to_stamped_pose_message(tmp, np.array([1., 0., 0., 0.]), "base_link")
            self.moveit_scene.add_box("right_wall", pose, [2.0, 0.001, 2.0])
            assert isec_utils.check_added_to_moveit_scene("robot_box", self.moveit_scene)

            tmp = np.copy(desk_center)
            tmp[1] += 0.17 + 0.2 + 0.1
            pose = isec_utils.to_stamped_pose_message(tmp, np.array([1., 0., 0., 0.]), "base_link")
            self.moveit_scene.add_box("left_wall", pose, [2.0, 0.001, 2.0])
            assert isec_utils.check_added_to_moveit_scene("robot_box", self.moveit_scene)

            tmp = np.copy(desk_center)
            tmp[0] += 0.17 + 0.1
            pose = isec_utils.to_stamped_pose_message(tmp, np.array([1., 0., 0., 0.]), "base_link")
            self.moveit_scene.add_box("far_wall", pose, [0.001, 2.0, 2.0])
            assert isec_utils.check_added_to_moveit_scene("robot_box", self.moveit_scene)

    def ur_script_move_to_j(self, joint_pos: Tuple[float, float, float, float, float, float], speed: float=0.5):

        assert not self.setup_planning, "Using move-it instead of ur script. Use plan_and_execute_joints_target or set setup_planning=False."

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
        if not plan_raw:
            raise exceptions.PlanningError()

        plan = plan_raw[1]
        success = self.moveit_move_group.execute(plan, wait=True)
        self.moveit_move_group.stop()
        self.moveit_move_group.clear_pose_targets()
        rospy.sleep(0.1)

        if not success:
            raise exceptions.ExecutionError()

    def plan_and_execute_pose_target_2(self, tool0_controller_pos, tool0_controller__quat):

        self.rviz_pub.send_pose(tool0_controller_pos, tool0_controller__quat, "base")            

        T = utils.pos_quat_to_transform(tool0_controller_pos, tool0_controller__quat)
        T = isec_utils.tool0_controller_base_to_flange_base_link(T, self.tf_proxy)
        base_link_to_flange_pos, base_link_to_flange_rot = utils.transform_to_pos_quat(T)

        assert self.setup_planning, "setup_planning has to be true."
        pose_msg = self.to_pose_message(base_link_to_flange_pos, base_link_to_flange_rot)
        self.setup_planning_attempt(pose_msg)

        # TODO: refactor
        num_plans = 10
        plans = []
        for i in range(num_plans):
            plan_raw = self.moveit_move_group.plan()
            if not plan_raw:
                # MoveIt silently wipes the planner when it fails to find a plan.
                self.setup_planning_attempt(pose_msg)
                continue

            plan = plan_raw[1]
            plans.append(plan)

        # TODO: test
        distances = []
        for plan in plans:
            distance = 0.
            prev_pos = None
            for point in plan.joint_trajectory.points:
                pos = np.array(point.positions)
                if prev_pos is not None:
                    distance += np.sum(np.abs(pos[:-1] - prev_pos[:-1]))
                    distance += 0.1 * np.abs(pos[-1] - prev_pos[-1])
                prev_pos = pos
            distances.append(distance)

        print("distances:", distances)
        idx = np.argmin(distances)
        distance = distances[idx]
        plan = plans[idx]

        if distance >= 10.:
            print("WARNING: The motion planner might be doing something weird.")
            input("Continue?")

        ds = DisplayTrajectory()
        ds.trajectory_start = self.moveit_robot.get_current_state()
        ds.trajectory.append(plan)
        self.rviz_pub.send_trajectory_message(ds)

        success = self.moveit_move_group.execute(plan, wait=True)
        self.moveit_move_group.stop()
        self.moveit_move_group.clear_pose_targets()
        rospy.sleep(0.1)

        if not success:
            raise exceptions.ExecutionError()

    def plan_and_execute_joints_target(self, joints):

        assert self.setup_planning, "setup_planning has to be true."
        self.moveit_move_group.set_max_velocity_scaling_factor(0.1)
        self.moveit_move_group.set_max_acceleration_scaling_factor(0.1)
        self.moveit_move_group.set_joint_value_target(joints)

        plan_raw = self.moveit_move_group.plan()
        if not plan_raw:
            raise exceptions.PlanningError()

        plan = plan_raw[1]
        success = self.moveit_move_group.execute(plan, wait=True)
        self.moveit_move_group.stop()
        self.moveit_move_group.clear_pose_targets()
        rospy.sleep(0.1)

        if not success:
            raise exceptions.ExecutionError()

    def setup_planning_attempt(self, pose_msg):

        self.moveit_move_group.set_num_planning_attempts(10)
        self.moveit_move_group.set_max_velocity_scaling_factor(0.1)
        self.moveit_move_group.set_max_acceleration_scaling_factor(0.1)
        self.moveit_move_group.set_pose_target(pose_msg)
        self.moveit_move_group.set_planner_id("RRTConnect")

    def joints_callback(self, msg: JointState):

        if self.joint_names_speedj[0] not in msg.name:
            return

        positions_dict = {}
        for i in range(len(msg.position)):
            positions_dict[msg.name[i]] = msg.position[i]
        self.joint_values = np.array([positions_dict[i] for i in self.joint_names_speedj])

    def get_end_effector_pose(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

        T = self.tf_proxy.lookup_transform("tool0_controller", "base")
        return utils.transform_to_pos_quat(T)

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
