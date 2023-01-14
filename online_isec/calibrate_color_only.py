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

    if args.camera == 0:
        frame = "cam1_color_optical_frame"
        gh, gw = (7, 9)
        gsize = 0.02
        gh, gw = (4, 5)
        gsize = 0.0384
    elif args.camera == 1:
        raise ValueError("Couldn't calibrate Azure.")
        # frame = "camera_depth_frame"
    elif args.camera == 2:
        frame = "depth_camera_link"
        gh, gw = (4, 5)
        gsize = 0.0384

    rot = Rotation.from_euler("z", -np.pi / 2.).as_matrix()
    offset = (constants.DESK_CENTER[0] + constants.WORKSPACE_SIZE / 2 - gh * gsize,
                constants.DESK_CENTER[1] - constants.WORKSPACE_SIZE / 2 + gw * gsize,
                constants.DESK_CENTER[2])

    topic = "/".join([*image_proxy.image_topics[args.camera].split("/")[:-1], "camera_info"])

    if args.camera != 2:
        # structure outputs IR image
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = ((image / image.max()) * 255.).astype(np.uint8)
        print(image.shape, image.dtype, image.max())

    plt.imshow(gray_image)
    plt.show()

    ret, corners = cv2.findChessboardCorners(gray_image, (gh, gw), None)
    assert ret

    if corners[0, :, 0] > corners[-1, :, 0]:
        print("Reversing detected corners to take care of 180deg symmetry of the checkerboard.")
        corners = corners[::-1]
    if args.camera == 2:
        corners = corners[::-1]

    show_corners(gray_image, corners)

    # TODO: center correctly, looking for the bottom-left corner of the workspace
    objp = np.zeros((gh * gw, 3), np.float32)
    tmp = gsize * np.dstack(np.mgrid[0: gw, 0: gh])
    print(tmp.shape)
    objp[:, :2] = tmp.reshape(-1, 2)

    objp = np.matmul(objp, rot.T)
    print(objp[..., 0].min(), objp[..., 0].max(), objp[..., 1].min(), objp[..., 1].max())

    # Take desk center, move to the bottom-right corner (robot's perspective),
    # move by either 7 or 9 checkboard squares.
    objp[:, 0] += offset[0]
    objp[:, 1] += offset[1]
    objp[:, 2] += offset[2]

    # plt.scatter(objp[:, 0], objp[:, 1], c=np.linspace(0, 1, objp.shape[0]))
    # plt.show()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray_image, np.array(corners), (5, 5), (-1, -1), criteria)
    show_corners(gray_image, corners2)

    camera_matrix, distortion_coeffs = isec_utils.get_camera_intrinsics_and_distortion(topic)

    ret, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, distortion_coeffs)
    # ret, _, _, rvec, tvec = cv2.calibrateCamera(objp[None], corners2[:, 0][None], (gray_image.shape[1], gray_image.shape[0]), camera_matrix, distortion_coeffs)
    # tvec = tvec[0]
    # rvec = rvec[0]

    tvec = tvec[:, 0]
    rvec = cv2.Rodrigues(rvec)[0]
    T = np.eye(4)
    T[:3, :3] = rvec
    T[:3, 3] = tvec
    print(T)
    T = np.linalg.inv(T)
    print(T)

    rvec = T[:3, :3]
    tvec = T[:3, 3]
    rvec = Rotation.from_matrix(rvec).as_quat()

    print(tvec)
    print(rvec)

    if args.publish:
        br = not_tensorflow.TransformBroadcaster()
        rate = rospy.Rate(50.0)

        while not rospy.is_shutdown():
            br.sendTransform(tvec, rvec, rospy.Time.now(), frame, "base")
            rate.sleep()


parser = argparse.ArgumentParser()
parser.add_argument("camera", type=int, help="0=realsense, 1=azure, 2=structure")
parser.add_argument("-p", "--publish", default=False, action="store_true")
main(parser.parse_args())
