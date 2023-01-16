import rospy
import cv2
import time

from online_isec.image_proxy import ImageProxy
import utils


def main():

    rospy.init_node("show_camera")

    image_proxy = ImageProxy()
    time.sleep(2)

    while True:
        image = image_proxy.get(0)[..., ::-1]
        if utils.update_opencv_window(image):
            break


main()
