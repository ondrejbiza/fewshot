import time
import rospy
import tf
from threading import Thread

from online_isec.point_cloud_proxy import RealsenseStructurePointCloudProxy
import viz_utils


def transmit(d):

    br = tf.TransformBroadcaster()
    rate = rospy.Rate(50.0)
  
    while not rospy.is_shutdown():
        br.sendTransform(d["realsense"][0], d["realsense"][1], rospy.Time.now(), "cam1_color_optical_frame", "base")
        br.sendTransform(d["structure"][0], d["structure"][1], rospy.Time.now(), "camera_depth_optical_frame", "base")
        rate.sleep()


def main():

    rospy.init_node("calibrate_by_hand")
    br = tf.TransformBroadcaster()

    sensor_d ={}
    sensor_d["realsense"] = [[-0.78302526, 0.0488187, 0.44769223], [-0.6538833, 0.66373159, -0.24695839, 0.26628662]]
    sensor_d["structure"] = [[-0.50859704, -0.55556215, 0.77484579], [0.65841246, 0.65057976, 0.262971, -0.27218609]]

    t = Thread(target=transmit, args=(sensor_d,))
    t.start()
    time.sleep(2)

    pc_proxy = RealsenseStructurePointCloudProxy()
    time.sleep(2)

    d = {
        "1": pc_proxy.get(0),
        "2": pc_proxy.get(1)
    }
    viz_utils.show_scene(d)


main()
