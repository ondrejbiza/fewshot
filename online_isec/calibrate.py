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

from online_isec.image_proxy import RealsenseStructureImageProxy
from online_isec.depth_proxy import RealsenseStructureDepthProxy
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
    image_proxy = RealsenseStructureImageProxy()
    depth_proxy = RealsenseStructureDepthProxy()
    time.sleep(2)

    image = image_proxy.get(args.camera)
    depth = depth_proxy.get(args.camera)
    assert image is not None and depth is not None

    image_proxy.close()
    depth_proxy.close()

    if args.camera == 1:
        image = (image / np.max(image)).astype(np.float32)

    topic = "/".join([*image_proxy.image_topics[args.camera].split("/")[:-1], "camera_info"])
    camera_matrix, distortion_coeffs = isec_utils.get_camera_intrinsics_and_distortion(topic)
    print(camera_matrix)
    print(np.linalg.inv(camera_matrix))

    if args.show:
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.subplot(2, 2, 3)
        plt.imshow(depth)

        # plt.subplot(4, 2, 2)
        # plt.imshow(0.5 * (image / 255.).astype(np.float32) + 0.5 * (depth / depth.max())[..., None])
        # plt.subplot(4, 2, 4)
        # plt.imshow((image / 255.).astype(np.float32) - (depth / depth.max())[..., None])

        plt.show()

    if args.camera == 0:
        frame = "cam1_color_optical_frame"
        gh, gw = (4, 5)
        gsize = 0.0384
    elif args.camera == 1:
        frame = "depth_camera_link"
        gh, gw = (4, 5)
        gsize = 0.0384
    else:
        raise ValueError("Invalid camera.")


    if args.camera == 1:
        # structure outputs IR image
        gray_image = ((image / image.max()) * 255.).astype(np.uint8)
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # plt.imshow(gray_image)
    # plt.show()

    ret, corners = cv2.findChessboardCorners(gray_image, (gh, gw), None)
    assert ret, "Did not find checkboard."

    if corners[0, :, 0] > corners[-1, :, 0]:
        print("Reversing detected corners to take care of 180deg symmetry of the checkerboard.")
        corners = corners[::-1]
    if args.camera == 1:
        corners = corners[::-1]

    if args.show:
        show_corners(gray_image, corners)

    # TODO: center correctly, looking for the bottom-left corner of the workspace
    objp = np.zeros((gh * gw, 3), np.float32)
    tmp = gsize * np.dstack(np.mgrid[0: gw, 0: gh])
    objp[:, :2] = tmp.reshape(-1, 2)

    rot = Rotation.from_euler("z", -np.pi / 2.).as_matrix()
    offset = (constants.DESK_CENTER[0] + constants.WORKSPACE_SIZE / 2 - gh * gsize,
                constants.DESK_CENTER[1] - constants.WORKSPACE_SIZE / 2 + gw * gsize,
                constants.DESK_CENTER[2])

    objp = np.matmul(objp, rot.T)
    objp[:, 0] += offset[0]
    objp[:, 1] += offset[1]
    objp[:, 2] += offset[2]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray_image, np.array(corners), (5, 5), (-1, -1), criteria)

    if args.show:
        show_corners(gray_image, corners2)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(image),
            o3d.geometry.Image(depth.astype(np.float32))
        ),
        o3d.camera.PinholeCameraIntrinsic(
            width=image.shape[1],
            height=image.shape[0],
            fx=camera_matrix[0, 0],
            fy=camera_matrix[1, 1],
            cx=camera_matrix[0, 2],
            cy=camera_matrix[1, 2]
        ),
        project_valid_depth_only=False
    )
    print("!!!!!", image.shape)
    pc = np.array(pcd.points, dtype=np.float32).reshape(image.shape[0], image.shape[1], 3)
    # mask = np.logical_not(np.any(np.isnan(pc), axis=-1))
    # pc = pc[mask]

    corners2 = corners2[:, 0, :]
    values = []
    for i in range(4):
        if i // 2 == 0:
            x = np.floor(corners2[:, 0]).astype(np.int32)
        else:
            x = np.ceil(corners2[:, 0]).astype(np.int32)
        if i % 2 == 0:
            y = np.floor(corners2[:, 1]).astype(np.int32)
        else:
            y = np.ceil(corners2[:, 1]).astype(np.int32)
        values.append(pc[y, x])
    values = np.mean(values, axis=0)
    print(values)

    T, r, t = utils.best_fit_transform(values, objp)

    tvec = t
    rvec = Rotation.from_matrix(r).as_quat()

    print(tvec)
    print(rvec)

    if args.show:

        out = values
        out = np.matmul(out, r.T)
        out = out + t

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(objp[..., 0], objp[..., 1], objp[..., 2])
        ax.scatter(out[..., 0], out[..., 1], out[..., 2])
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(-0.2, 0.2)
        plt.show()

    if args.publish:
        br = not_tensorflow.TransformBroadcaster()
        rate = rospy.Rate(50.0)

        while not rospy.is_shutdown():
            br.sendTransform(tvec, rvec, rospy.Time.now(), frame, "base")
            rate.sleep()

    return
    # camera_matrix, distortion_coeffs = isec_utils.get_camera_intrinsics_and_distortion(topic)

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



parser = argparse.ArgumentParser()
parser.add_argument("camera", type=int, help="0=realsense, 1=structure")
parser.add_argument("-p", "--publish", default=False, action="store_true")
parser.add_argument("-s", "--show", default=False, action="store_true")
main(parser.parse_args())
