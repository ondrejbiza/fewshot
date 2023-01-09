from typing import Tuple
import time
import rospy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import cv2
from scipy.spatial.transform import Rotation

from online_isec.point_cloud_proxy import PointCloudProxy
import utils


def main():

    rospy.init_node("calibrate")
    pc_proxy = PointCloudProxy()
    time.sleep(2)

    _, image, _ = pc_proxy.get(0)
    assert image is not None

    gh, gw = (9, 7)
    gsize = 0.02

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_image, (gh, gw), None)
    assert ret

    objp = np.zeros((gh * gw, 3), np.float32)
    objp[:, :2] = gsize * np.dstack(np.mgrid[1: gw + 1, -1: gh - 1]).reshape(-1, 2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray_image, np.array(corners), (11, 11), (-1, -1), criteria)

    camera_matrix = np.array([
        632.2827758789062, 0.0, 636.6957397460938, 0.0, 0.0, 631.7894287109375, 361.6166076660156, 0.0, 0.0, 0.0, 1.0, 0.0
    ], dtype=np.float32).reshape(3, 4)[:, :3]
    distortion_coeffs = np.array([
        -0.056411661207675934, 0.06755722314119339, -0.0007080861832946539, 0.00018199955229647458, -0.021540766581892967
    ], dtype=np.float32)

    ret, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, distortion_coeffs)
    rvec = cv2.Rodrigues(rvec)[0]
    rvec = Rotation.from_matrix(rvec).as_quat()
    print(tvec)
    print(rvec)


main()
