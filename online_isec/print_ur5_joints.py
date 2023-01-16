import rospy
import time

from online_isec.ur5 import UR5


def main():
    rospy.init_node("ur5_control")
    ur5 = UR5()
    time.sleep(2)
    print(ur5.joint_values)


main()
