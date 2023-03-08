from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from moveit_msgs.msg import DisplayTrajectory
import moveit_commander
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from src import exceptions, utils
import src.real_world.constants as rw_constants
import src.real_world.utils as rw_utils
from src.real_world.moveit_scene import MoveItScene
from src.real_world.robotiq_gripper import RobotiqGripper
from src.real_world.rviz_pub import RVizPub
from src.real_world.tf_proxy import TFProxy


@dataclass
class UR5:

    activate_gripper: bool = True

    pick_offset: float = 0.1
    place_offset: float = 0.1
    place_open_pos: float = 0.

    home_joint_values: Tuple[float, ...] = (0.65601951, -1.76965791, 1.79728603, -1.60219127, -1.5338834, -0.785)
    desk_joint_values: Tuple[float, ...] = (-0.1835673491, -1.446624104, 1.77286005, -1.90146142, -1.532696072, 1.339956641)
    joint_names_speedj: Tuple[str, ...] = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                                           "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    
    setup_planning: bool = False
    move_group_name: str = "manipulator"
    tool_frame_id: str = "flange"

    def __post_init__(self):

        self.tf_proxy = TFProxy()
        self.rviz_pub = RVizPub()

        self.gripper = RobotiqGripper()
        if self.activate_gripper:
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
            self.moveit_scene = MoveItScene()
            self.moveit_move_group = moveit_commander.MoveGroupCommander(self.move_group_name)
            self.moveit_move_group.set_end_effector_link(self.tool_frame_id)
            rospy.sleep(2)

            self.moveit_scene.clear()

            desk_center = utils.transform_pcd(np.array(rw_constants.DESK_CENTER)[None], self.tf_proxy.lookup_transform("base", "base_link"))[0]
            self.moveit_scene.add_box("table", desk_center, np.array([1., 0., 0., 0.]), np.array([2., 2., 0.001]))

            self.moveit_scene.add_box("robot_box", np.array([0., 0., -0.04]), np.array([1., 0., 0., 0.]), np.array([0.16, 0.5, 0.075]))

            tmp = np.copy(desk_center)
            tmp[1] -= 0.17 + 0.2 + 0.1
            self.moveit_scene.add_box("right_wall", tmp, np.array([1., 0., 0., 0.]), np.array([2.0, 0.001, 2.0]))

            tmp = np.copy(desk_center)
            tmp[1] += 0.17 + 0.2 + 0.2
            self.moveit_scene.add_box("left_wall", tmp, np.array([1., 0., 0., 0.]), np.array([2.0, 0.001, 2.0]))

            tmp = np.copy(desk_center)
            tmp[0] += 0.17 + 0.1
            self.moveit_scene.add_box("far_wall", tmp, np.array([1., 0., 0., 0.]), np.array([0.001, 2.0, 2.0]))

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

    def plan_and_execute_pose_target(self, tool0_pos, tool0_quat, num_plans: int=10):

        self.rviz_pub.send_pose(tool0_pos, tool0_quat, "base")            

        plans = []

        T = utils.pos_quat_to_transform(tool0_pos, tool0_quat)
        T = rw_utils.tool0_base_to_flange_base_link(T, self.tf_proxy)
        base_link_to_flange_pos, base_link_to_flange_rot = utils.transform_to_pos_quat(T)

        assert self.setup_planning, "setup_planning has to be true."
        pose_msg = rw_utils.to_pose_message(base_link_to_flange_pos, base_link_to_flange_rot)
        self.setup_planning_attempt(pose_msg)

        for i in range(num_plans):
            plan_raw = self.moveit_move_group.plan()
            if not plan_raw[0]:
                # not sure if I need this anymore.
                self.setup_planning_attempt(pose_msg)
                continue

            plan = plan_raw[1]
            plans.append(plan)

        if len(plans) == 0:
            raise exceptions.PlanningError("Could not find a plan.")

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

        print("Plan distances in joint space:", distances)
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

    def get_end_effector_pose_2(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Move from tool0 to the tip manually. tool0_controller is miscalibrated."""
        trans_t0_to_b = self.tf_proxy.lookup_transform("tool0", "base")
        trans_t0_tip_to_b = trans_t0_to_b @ rw_utils.tool0_tip_to_tool0()
        return utils.transform_to_pos_quat(trans_t0_tip_to_b)


    def get_tool0_to_base(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

        T = self.tf_proxy.lookup_transform("tool0", "base")
        return utils.transform_to_pos_quat(T)
