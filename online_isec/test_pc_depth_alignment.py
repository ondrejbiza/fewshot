import time
import rospy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

from online_isec.point_cloud_proxy import RealsenseStructurePointCloudProxy
from online_isec.depth_proxy import RealsenseStructureDepthProxy
from online_isec.image_proxy import RealsenseStructureImageProxy
import online_isec.utils as isec_utils
import utils
import viz_utils


def main():

    rospy.init_node("test_pc_depth_alignment")
    pc_proxy = RealsenseStructurePointCloudProxy(apply_transform=False)
    depth_proxy = RealsenseStructureDepthProxy()
    image_proxy = RealsenseStructureImageProxy()
    time.sleep(2)

    camera_index = 1
    cloud = pc_proxy.get(camera_index)
    image = image_proxy.get(camera_index)
    depth = depth_proxy.get(camera_index)
    assert cloud is not None and image is not None and depth is not None

    pc_proxy.close()
    depth_proxy.close()
    image_proxy.close()

    if camera_index == 1:
        image = (image / image.max()).astype(np.float32)

    topic = "/".join([*image_proxy.image_topics[camera_index].split("/")[:-1], "camera_info"])
    camera_matrix, distortion_coeffs = isec_utils.get_camera_intrinsics_and_distortion(topic)

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
    pc = np.array(pcd.points, dtype=np.float32)
    mask = np.logical_not(np.any(np.isnan(pc), axis=-1))
    pc = pc[mask]

    print(pc.shape, cloud.shape)

    viz_utils.show_scene({
        "1": pc,
        "2": cloud
    })


main()
