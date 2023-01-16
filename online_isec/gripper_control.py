import rospy

from online_isec.ur5 import UR5


def main():
    rospy.init_node("gripper_control")
    ur5 = UR5()

    while True:
        inp = input("Gripper o/c: ")
        if inp == "o":
            ur5.gripper.open_gripper()
        elif inp == "c":
            ur5.gripper.close_gripper()


main()
