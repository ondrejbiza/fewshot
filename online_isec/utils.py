import time
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import rospy
from sensor_msgs.msg import CameraInfo
import open3d as o3d


def get_camera_intrinsics_and_distortion(topic: str) -> Tuple[NDArray, NDArray]:

    out = [False, None, None]
    def callback(msg: CameraInfo):
        out[1] = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        out[2] = np.array(msg.D, dtype=np.float64)
        out[0] = True
    
    sub = rospy.Subscriber(topic, CameraInfo, callback, queue_size=1)
    for _ in range(100):
        time.sleep(0.1)
        if out[0]:
            sub.unregister()
            return out[1], out[2]

    raise RuntimeError("Could not get camera information.")


def mask_workspace(cloud: NDArray, desk_center: Tuple[float, float, float], size: float=0.2) -> NDArray:

    cloud = np.copy(cloud)
    cloud[..., 0] -= desk_center[0]
    cloud[..., 1] -= desk_center[1]
    cloud[..., 2] -= desk_center[2]

    mask = np.logical_and(np.abs(cloud[..., 0]) <= size, np.abs(cloud[..., 1]) <= size)
    mask = np.logical_and(mask, cloud[..., 2] >= 0.)
    mask = np.logical_and(mask, cloud[..., 2] <= 2 * size)

    return cloud[mask]


def find_mug_and_tree(cloud: NDArray) -> Tuple[NDArray, NDArray]:

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))

    pcs = []
    for label in np.unique(labels):
        if label == -1:
            # Background label?
            continue
        
        pc = cloud[labels == label]
        if np.min(pc[..., 2]) > 0.1:
            # Above ground, probably robot gripper.
            continue

        pcs.append(pc)

    assert len(pcs) == 2, "The world must have exactly two objects."
    assert len(pcs[0]) > 10, "Too small PC."
    assert len(pcs[1]) > 10, "Too small PC."

    # Tree is taller than mug.
    if np.max(pcs[0][..., 2]) > np.max(pcs[1][..., 2]):
        tree = pcs[0]
        mug = pcs[1]
    else:
        tree = pcs[1]
        mug = pcs[0]
    
    # Cut off the base of the tree.
    mask = tree[..., 2] >= 0.03
    tree = tree[mask]

    return mug, tree
