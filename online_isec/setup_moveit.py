import time
import rospy
import moveit_commander
import numpy as np

from online_isec.tf_proxy import TFProxy
import utils


def main():

    rospy.init_node("moveit_plan_pick")
    time.sleep(1)

    name = "manipulator"
    tool_frame_id = "flange"

    tf_proxy = TFProxy()
    T_1 = tf_proxy.lookup_transform("flange", "base_link")
    T_2 = tf_proxy.lookup_transform("tool0_controller", "base")
    T = utils.compute_relative_transform(T_1, T_2)

    print(T_1)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    move_group = moveit_commander.MoveGroupCommander(name)
    move_group.set_end_effector_link(tool_frame_id)
    
    flange_pose = move_group.get_current_pose()
    flange_pos = np.array([flange_pose.pose.position.x, flange_pose.pose.position.y, flange_pose.pose.position.z], dtype=np.float32)
    flange_quat = np.array([flange_pose.pose.orientation.x, flange_pose.pose.orientation.y, flange_pose.pose.orientation.z, flange_pose.pose.orientation.w], dtype=np.float32)

    print(flange_pos, flange_quat)


main()
