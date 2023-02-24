from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import open3d as o3d

from src import utils
from src.real_world import constants


def center_workspace(cloud: NDArray, desk_center: Tuple[float, float, float]=constants.DESK_CENTER) -> NDArray:

    cloud = np.copy(cloud)
    cloud[..., 0] -= desk_center[0]
    cloud[..., 1] -= desk_center[1]
    cloud[..., 2] -= desk_center[2]
    return cloud


def mask_workspace(cloud: NDArray, size: float=constants.WORKSPACE_SIZE, height_eps: float=constants.HEIGHT_EPS) -> NDArray:

    half_size = size // 2

    mask = np.logical_and(np.abs(cloud[..., 0]) <= half_size, np.abs(cloud[..., 1]) <= half_size)
    mask = np.logical_and(mask, cloud[..., 2] >= height_eps)
    mask = np.logical_and(mask, cloud[..., 2] <= size)

    return cloud[mask]


def find_mug_and_tree(cloud: NDArray, tall_mug_plaform: bool=False, short_mug_platform: bool=False) -> Tuple[NDArray, NDArray]:

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    labels = np.array(pcd.cluster_dbscan(eps=0.03, min_points=10))

    print("PC lengths (ignoring PCs above the ground).")
    pcs = []
    for label in np.unique(labels):
        if label == -1:
            # Background label?
            continue
        
        pc = cloud[labels == label]
        if np.min(pc[..., 2]) > 0.1:
            # Above ground, probably robot gripper.
            continue

        print(len(pc))

        pcs.append(pc)

    assert len(pcs) >= 2, "The world must have at least two objects."
    if len(pcs) > 2:
        # Pick the two point clouds with the most points.
        sizes = [len(pc) for pc in pcs]
        sort = list(reversed(np.argsort(sizes)))
        pcs = [pcs[sort[0]], pcs[sort[1]]]

    assert len(pcs[0]) > 10, "Too small PC."
    assert len(pcs[1]) > 10, "Too small PC."

    # Tree is taller than mug.
    if np.max(pcs[0][..., 2]) > np.max(pcs[1][..., 2]):
        tree = pcs[0]
        mug = pcs[1]
    else:
        tree = pcs[1]
        mug = pcs[0]

    if tall_mug_plaform:
        mug = mug[mug[..., 2] > 0.14]

    if short_mug_platform:
        # mug = mug[mug[..., 2] > 0.02]
        mug = mug[mug[..., 2] > 0.13]

    # Cut off the base of the tree.
    # No base.
    mask = tree[..., 2] >= 0.03
    # With base.
    # mask = tree[..., 2] >= 0.05
    tree = tree[mask]

    return mug, tree


def mug_tree_simple_perception(cloud: NDArray, max_pc_size: Optional[int]=2000) -> Tuple[NDArray, NDArray]:

    cloud = center_workspace(cloud)
    cloud = mask_workspace(cloud)
    # viz_utils.o3d_visualize(utils.create_o3d_pointcloud(cloud))

    mug_pc, tree_pc = find_mug_and_tree(cloud)
    if max_pc_size is not None:
        if len(mug_pc) > max_pc_size:
            mug_pc, _ = utils.farthest_point_sample(mug_pc, max_pc_size)
        if len(tree_pc) > max_pc_size:
            tree_pc, _ = utils.farthest_point_sample(tree_pc, max_pc_size)

    return mug_pc, tree_pc
