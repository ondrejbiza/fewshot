import argparse
import rospy

from src.real_world.ur5 import UR5


def main():
    rospy.init_node("ur5_control")
    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


parser = argparse.ArgumentParser("Move UR5 to home position.")
main()
