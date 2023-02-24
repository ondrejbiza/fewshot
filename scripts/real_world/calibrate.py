import argparse
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import rospy
from scipy.spatial.transform import Rotation

from src import viz_utils
import src.real_world.utils as rw_utils
from src.real_world.image_proxy import ImageProxy


def show_corners(gray_image: NDArray, corners: NDArray):

    img = np.array(gray_image)
    for i, corner in enumerate(corners):
        x, y = int(corner[0, 1]), int(corner[0, 0])
        viz_utils.draw_square(img, x, y, square_size=6, intensity=i / len(corners))
    plt.imshow(img)
    plt.show()


def main(args):

    rospy.init_node("calibrate")
    image_proxy = ImageProxy()
    time.sleep(2)

    # Height and width of the checkerboard minus one (the number of internal corners).
    # Size of a single tile.
    if args.transpose:
        gh, gw = (5, 4)
    else:
        gh, gw = (4, 5)
    gsize = 0.0384

    # Camera frame names.
    if args.camera == 0:
        frame = "realsense_left_color_optical_frame"
    elif args.camera == 1:
        frame = "realsense_right_color_optical_frame"
    elif args.camera == 2:
        frame = "realsense_forward_color_optical_frame"
    else:
        raise ValueError("Invalid camera index.")

    # This topic should give us the intrinsic camera matrix and distortion coefficients.
    topic = "/".join([*image_proxy.image_topics[args.camera].split("/")[:-1], "camera_info"])

    for _ in range(100):

        image = image_proxy.get(args.camera)
        assert image is not None
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if args.show:
            plt.imshow(gray_image)
            plt.show()

        # Find checkerboard corners.
        ret, corners = cv2.findChessboardCorners(
            gray_image, 
            (gh, gw),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

        print(ret)
        if ret:
            break

    assert ret

    if args.show:
        show_corners(gray_image, corners)

    # Refine checkerboard corners.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray_image, np.array(corners), (5, 5), (-1, -1), criteria)
    if args.show:
        show_corners(gray_image, corners2)

    # Create a coordinate grid of the corners of the checkboard.
    objp = np.zeros((gh * gw, 3), np.float32)
    tmp = gsize * np.dstack(np.mgrid[0: gw, 0: gh])
    objp[:, :2] = tmp.reshape(-1, 2)

    # Calculate extrinsic camera matrix.
    camera_matrix, distortion_coeffs = rw_utils.get_camera_intrinsics_and_distortion(topic)
    ret, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, distortion_coeffs)

    # Transform to matrix, invert, transform to translation and quaternions.
    tvec = tvec[:, 0]
    rvec = cv2.Rodrigues(rvec)[0]
    T = np.eye(4)
    T[:3, :3] = rvec
    T[:3, 3] = tvec

    T = np.linalg.inv(T)
    rvec = T[:3, :3]
    tvec = T[:3, 3]
    rvec = Rotation.from_matrix(rvec).as_quat()

    print("Translation:", tvec)
    print("Rotation (quat):", rvec)
    print('br.sendTransform(({:f}, {:f}, {:f}), ({:f}, {:f}, {:f}, {:f}), rospy.Time.now(), "{:s}", "checkerboard")'.format(
        tvec[0], tvec[1], tvec[2], rvec[0], rvec[1], rvec[2], rvec[3], frame
    ))



parser = argparse.ArgumentParser("Calibrate cameras using a 5x4 checkerboard.")
parser.add_argument("camera", type=int, help="0=realsense_left, 1=realsense_right, 2=realsense_forward")
parser.add_argument("-s", "--show", default=False, action="store_true")
parser.add_argument("-t", "--transpose", default=False, action="store_true")
main(parser.parse_args())
