import numpy as np
import rospy
import time

from online_isec import constants
from online_isec.point_cloud_proxy import RealsenseStructurePointCloudProxy
from online_isec import perception


def main():

    rospy.init_node("easy_perception")
    pc_proxy = RealsenseStructurePointCloudProxy()
    time.sleep(2)

    perception.mug_tree_perception(
        pc_proxy, np.array(constants.DESK_CENTER), mug_save_decomposition=False
    )


main()
