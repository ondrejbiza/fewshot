import copy as cp
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import torch
from scipy.spatial import KDTree

from src import object_warping, utils, viz_utils
import src.real_world.utils as rw_utils
from src.real_world import constants
from src.real_world.tf_proxy import TFProxy
from src.real_world.moveit_scene import MoveItScene
from src.real_world.rviz_pub import RVizPub

NPF32 = NDArray[np.float32]
NPF64 = NDArray[np.float64]


def center_workspace(cloud: NPF32, desk_center: Tuple[float, float, float]=constants.DESK_CENTER) -> NPF32:

    cloud = np.copy(cloud)
    cloud[..., 0] -= desk_center[0]
    cloud[..., 1] -= desk_center[1]
    cloud[..., 2] -= desk_center[2]
    return cloud


def mask_workspace(cloud: NPF32, size: float=constants.WORKSPACE_SIZE, height_eps: float=constants.HEIGHT_EPS) -> NDArray:

    half_size = size / 2

    mask = np.logical_and(np.abs(cloud[..., 0]) <= half_size, np.abs(cloud[..., 1]) <= half_size)
    mask = np.logical_and(mask, cloud[..., 2] >= height_eps)
    mask = np.logical_and(mask, cloud[..., 2] <= size)

    return cloud[mask]


def subtract_platform(source_pcd: NPF32, platform_pcd: NPF32, delta: float=0.01) -> NPF32:
    """Subtract platform point cloud from source object point cloud."""
    # KDTrees for faster lookup.
    tree_platform = KDTree(platform_pcd)
    tree_source = KDTree(source_pcd)
    mask_lists = tree_source.query_ball_tree(tree_platform, r=delta)
    assert len(mask_lists) == len(source_pcd)

    # Remove all points that are at most delta away from any point in the platform point cloud.
    mask = np.ones(len(source_pcd), dtype=np.bool_)
    for i in range(len(mask_lists)):
        if len(mask_lists[i]) > 0:
            mask[i] = False

    return source_pcd[mask]


def find_mug_and_tree(cloud: NPF32) -> Tuple[NPF32, NPF32]:

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

    # Cut off the base of the tree.
    # No base.
    mask = tree[..., 2] >= 0.03
    # With base.
    # mask = tree[..., 2] >= 0.05
    tree = tree[mask]

    return mug, tree


def find_bowl_and_mug(cloud: NPF32) -> Tuple[NPF32, NPF32]:

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

    # Bowl is wider than mug.
    var1 = np.mean(np.sum(np.square(pcs[0] - np.mean(pcs[0], axis=0, keepdims=True)), axis=-1))
    var2 = np.mean(np.sum(np.square(pcs[1] - np.mean(pcs[1], axis=0, keepdims=True)), axis=-1))

    if var1 > var2:
        bowl = pcs[0]
        mug = pcs[1]
    else:
        bowl = pcs[1]
        mug = pcs[0]

    return bowl, mug


def find_bottle_and_box(cloud: NPF32) -> Tuple[NPF32, NPF32]:

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

    # Bottle is taller than box.
    if np.max(pcs[0][..., 2]) > np.max(pcs[1][..., 2]):
        bottle = pcs[0]
        box = pcs[1]
    else:
        bottle = pcs[1]
        box = pcs[0]

    return bottle, box


def mug_tree_segmentation(cloud: NPF32, max_pc_size: Optional[int]=2000,
                          platform_pcd: Optional[NPF32]=None) -> Tuple[NPF32, NPF32]:

    cloud = center_workspace(cloud)
    cloud = mask_workspace(cloud)

    mug_pcd, tree_pcd = find_mug_and_tree(cloud)
    if platform_pcd is not None:
        mug_pcd = subtract_platform(mug_pcd, platform_pcd)

    if max_pc_size is not None:
        if len(mug_pcd) > max_pc_size:
            mug_pcd, _ = utils.farthest_point_sample(mug_pcd, max_pc_size)
        if len(tree_pcd) > max_pc_size:
            tree_pcd, _ = utils.farthest_point_sample(tree_pcd, max_pc_size)

    return mug_pcd, tree_pcd


def bowl_mug_segmentation(cloud: NPF32, max_pc_size: Optional[int]=2000,
                          platform_pcd: Optional[NPF32]=None) -> Tuple[NPF32, NPF32]:

    cloud = center_workspace(cloud)
    cloud = mask_workspace(cloud)

    bowl_pcd, mug_pcd = find_bowl_and_mug(cloud)
    if platform_pcd is not None:
        bowl_pcd = subtract_platform(bowl_pcd, platform_pcd)

    if max_pc_size is not None:
        if len(bowl_pcd) > max_pc_size:
            bowl_pcd, _ = utils.farthest_point_sample(bowl_pcd, max_pc_size)
        if len(mug_pcd) > max_pc_size:
            mug_pcd, _ = utils.farthest_point_sample(mug_pcd, max_pc_size)

    return bowl_pcd, mug_pcd


def bottle_box_segmentation(cloud: NPF32, max_pc_size: Optional[int]=2000,
                            platform_pcd: Optional[NPF32]=None) -> Tuple[NPF32, NPF32]:

    cloud = center_workspace(cloud)
    cloud = mask_workspace(cloud)

    bottle_pcd, box_pcd = find_bottle_and_box(cloud)
    if platform_pcd is not None:
        bottle_pcd = subtract_platform(bottle_pcd, platform_pcd)

    if max_pc_size is not None:
        if len(bottle_pcd) > max_pc_size:
            bottle_pcd, _ = utils.farthest_point_sample(bottle_pcd, max_pc_size)
        if len(box_pcd) > max_pc_size:
            box_pcd, _ = utils.farthest_point_sample(box_pcd, max_pc_size)

    return bottle_pcd, box_pcd


def platform_segmentation(cloud: NPF32) -> NPF32:
    """Segment platform from the table."""
    cloud = center_workspace(cloud)
    cloud = mask_workspace(cloud)
    return cloud


def warping(
    source_pcd: NPF32, target_pcd: NPF32,
    canon_source: utils.CanonObj, canon_target: utils.CanonObj,
    tf_proxy: Optional[TFProxy]=None,
    moveit_scene: Optional[MoveItScene]=None,
    source_save_decomposition: bool=False,
    target_save_decomposition: bool=False,
    add_source_to_planning_scene: bool=False,
    add_target_to_planning_scene: bool=False,
    rviz_pub: Optional[RVizPub]=None,
    ablate_no_warping: bool=False,
    source_any_rotation: bool=False,
    source_no_warping: bool=False,
    desk_center: Tuple[float, float, float]=constants.DESK_CENTER,
    grow_source_object: bool=False, grow_target_object: bool=False
    ) -> Tuple[NPF32, utils.ObjParam, NPF32, utils.ObjParam, utils.CanonObj, utils.CanonObj, NPF32, NPF32]:

    if ablate_no_warping:
        raise NotImplementedError()

    dc = np.array(desk_center)
    perception_start = time.time()

    if source_any_rotation:
        assert not source_no_warping
        warp = object_warping.ObjectWarpingSE3Batch(
            canon_source, source_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=1000, object_size_reg=0.01, scaling=True, init_scale=canon_source.init_scale)
        source_pcd_complete, _, source_param = object_warping.warp_to_pcd_se3(warp, n_angles=12, n_batches=15)
    else:
        if source_no_warping:
            warp = object_warping.ObjectSE2Batch(
                canon_source, source_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
                n_samples=1000, scaling=False, init_scale=canon_source.init_scale)
            source_pcd_complete, _, source_param = object_warping.warp_to_pcd_se2(warp, n_angles=12, n_batches=1)
        else:
            warp = object_warping.ObjectWarpingSE2Batch(
                canon_source, source_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
                n_samples=1000, object_size_reg=0.01, scaling=True, init_scale=canon_source.init_scale)
            source_pcd_complete, _, source_param = object_warping.warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

    warp = object_warping.ObjectWarpingSE2Batch(
        canon_target, target_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
        n_samples=1000, object_size_reg=0.01, scaling=True, init_scale=canon_target.init_scale)
    target_pcd_complete, _, target_param = object_warping.warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

    print("Inference time: {:.1f}s.".format(time.time() - perception_start))

    viz_utils.show_pcds_plotly({
        "source": source_pcd,
        "source_complete": source_pcd_complete,
        "target": target_pcd,
        "target_complete": target_pcd_complete,
    })

    if grow_source_object:
        tmp = cp.deepcopy(source_param)
        tmp.scale *= 1.2
        mug_mesh = canon_source.to_mesh(tmp)
    else:
        mug_mesh = canon_source.to_mesh(source_param)

    mug_mesh.export("tmp_source.stl")
    if source_save_decomposition:
        utils.convex_decomposition(mug_mesh, "tmp_source.obj")

    if grow_target_object:
        tmp = cp.deepcopy(target_param)
        tmp.scale *= 1.25
        tree_mesh = canon_target.to_mesh(tmp)
    else:
        tree_mesh = canon_target.to_mesh(target_param)

    tree_mesh.export("tmp_target.stl")
    if target_save_decomposition:
        utils.convex_decomposition(tree_mesh, "tmp_target.obj")

    if add_source_to_planning_scene:
        assert moveit_scene is not None and tf_proxy is not None, "Need moveit_scene and tf_proxy to add an object to the planning scene."
        pos, quat = utils.transform_to_pos_quat(rw_utils.desk_obj_param_to_base_link_T(source_param.position, source_param.quat, dc, tf_proxy))
        moveit_scene.add_object("tmp_source.stl", "mug", pos, quat)

    if add_target_to_planning_scene:
        assert moveit_scene is not None and tf_proxy is not None, "Need moveit_scene and tf_proxy to add an object to the planning scene."
        pos, quat = utils.transform_to_pos_quat(rw_utils.desk_obj_param_to_base_link_T(target_param.position, target_param.quat, dc, tf_proxy))
        moveit_scene.add_object("tmp_target.stl", "tree", pos, quat)

    if rviz_pub is not None:
        # send STL meshes as Marker messages to rviz
        assert tf_proxy is not None, "Need tf_proxy to calculate poses."

        pos, quat = utils.transform_to_pos_quat(rw_utils.desk_obj_param_to_base_link_T(source_param.position, source_param.quat, dc, tf_proxy))
        rviz_pub.send_stl_message("tmp_source.stl", pos, quat)

        pos, quat = utils.transform_to_pos_quat(rw_utils.desk_obj_param_to_base_link_T(target_param.position, target_param.quat, dc, tf_proxy))
        rviz_pub.send_stl_message("tmp_target.stl", pos, quat)

    return source_pcd_complete, source_param, target_pcd_complete, target_param, canon_source, canon_target, source_pcd, target_pcd
