import argparse
import rospy
import time
import numpy as np

from online_isec.ur5 import UR5


def main(args):
    rospy.init_node("ur5_control")
    ur5 = UR5(setup_planning=True, move_group_name=args.group_name, tool_frame_id=args.tool_name)
    input("big red button")
    ur5.plan_and_execute_pose_target(np.array([0., 0., 0.]), np.array([1., 0., 0., 0.]))  # TODO: wrong unit quats everywhere!


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group_name")
    parser.add_argument("tool_name")
    main(parser.parse_args())
