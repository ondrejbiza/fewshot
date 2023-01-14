import argparse
from typing import Tuple
import time
import rospy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import cv2
from scipy.spatial.transform import Rotation
import tf as not_tensorflow
from sensor_msgs.msg import CameraInfo

from online_isec.image_proxy import ImageProxy
from online_isec import constants
import utils
import viz_utils
from online_isec import utils as isec_utils
from online_isec import constants


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

    image = image_proxy.get(args.camera)
    assert image is not None

    # Height and width of the checkerboard minus one (the number of internal corners).
    # Size of a single tile.
    gh, gw = (4, 5)
    gsize = 0.0384

    # Transform that goes from the coordinate frame of the checkerboard to the coordinate frame of the base.
    # !!! I am using hacks so that my three cameras detect the checkerboard corners in the same order.
    rot = Rotation.from_euler("z", -np.pi / 2.).as_matrix()
    offset = (constants.DESK_CENTER[0] + constants.WORKSPACE_SIZE / 2 - gh * gsize,
              constants.DESK_CENTER[1] - constants.WORKSPACE_SIZE / 2 + gw * gsize,
              constants.DESK_CENTER[2])
    if args.camera == 1:
        offset = (offset[0], offset[1] + 0.1696, offset[2])

    # Camera frame names.
    if args.camera == 0:
        frame = "cam1_color_optical_frame"
    elif args.camera == 1:
        frame = "depth_camera_link"
    elif args.camera == 2:
        frame = "camera_depth_optical_frame"
    else:
        raise ValueError("Invalid camera index.")

    # This topic should give us the intrinsic camera matrix and distortion coefficients.
    topic = "/".join([*image_proxy.image_topics[args.camera].split("/")[:-1], "camera_info"])

    if args.camera == 2:
        # Structure outputs IR image. Normalize to [0, 1].
        gray_image = ((image / image.max()) * 255.).astype(np.uint8)
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if args.show:
        plt.imshow(gray_image)
        plt.show()

    # Find checkerboard corners.
    ret, corners = cv2.findChessboardCorners(gray_image, (gh, gw), None)
    assert ret

    # Hacks so that they are always in the same order.
    if corners[0, :, 0] > corners[-1, :, 0]:
        print("Reversing detected corners to take care of 180deg symmetry of the checkerboard.")
        corners = corners[::-1]
    if args.camera == 2:
        corners = corners[::-1]
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

    # Transform from checkboard frame to base frame.
    # We could have an automatic method of doing this, but then we'd need to move the robot
    # and select where it is in the image. Or something like that ...
    # Instead, I know exactly where I placed the checkerboard w.r.t. the base.
    objp = np.matmul(objp, rot.T)
    objp[:, 0] += offset[0]
    objp[:, 1] += offset[1]
    objp[:, 2] += offset[2]

    # Calculate extrinsic camera matrix.
    camera_matrix, distortion_coeffs = isec_utils.get_camera_intrinsics_and_distortion(topic)
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
    print('br.sendTransform(({:f}, {:f}, {:f}), ({:f}, {:f}, {:f}, {:f}), rospy.Time.now(), "{:s}", "base")'.format(
        tvec[0], tvec[1], tvec[2], rvec[0], rvec[1], rvec[2], rvec[3], frame
    ))

    if args.publish:
        br = not_tensorflow.TransformBroadcaster()
        rate = rospy.Rate(50.0)

        while not rospy.is_shutdown():
            br.sendTransform(tvec, rvec, rospy.Time.now(), frame, "base")
            rate.sleep()


parser = argparse.ArgumentParser()
parser.add_argument("camera", type=int, help="0=realsense, 1=azure, 2=structure")
parser.add_argument("-p", "--publish", default=False, action="store_true")
parser.add_argument("-s", "--show", default=False, action="store_true")
main(parser.parse_args())
