import rospy
import time

from online_isec.ur5 import UR5


def main():
    rospy.init_node("ur5_control")
    ur5 = UR5()
    time.sleep(2)
    ur5.move_to_j(ur5.home_joint_values)


main()
