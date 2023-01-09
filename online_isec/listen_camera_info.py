import rospy
import ros_numpy
from sensor_msgs.msg import CameraInfo
import time


def callback(msg):

    print(msg)


def main():

    rospy.init_node("listen_camera_info")
    time.sleep(2)

    topic = "/cam1/color/camera_info"
    sub = rospy.Subscriber(topic, CameraInfo, callback, queue_size=1)
    while True:
        time.sleep(1)


main()
