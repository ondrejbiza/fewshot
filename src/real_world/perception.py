import copy as cp
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import torch

from src import object_warping, utils, viz_utils
import src.real_world.utils as rw_utils
from src.real_world import constants
from src.real_world.tf_proxy import TFProxy
from src.real_world.moveit_scene import MoveItScene
from src.real_world.rviz_pub import RVizPub


def center_workspace(cloud: NDArray, desk_center: Tuple[float, float, float]=constants.DESK_CENTER) -> NDArray:

    cloud = np.copy(cloud)
    cloud[..., 0] -= desk_center[0]
    cloud[..., 1] -= desk_center[1]
    cloud[..., 2] -= desk_center[2]
    return cloud


def mask_workspace(cloud: NDArray, size: float=constants.WORKSPACE_SIZE, height_eps: float=constants.HEIGHT_EPS) -> NDArray:

    half_size = size / 2

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


def find_bowl_and_mug(cloud: NDArray, platform_1: bool=False) -> Tuple[NDArray, NDArray]:

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

    if platform_1:
        bowl = bowl[bowl[..., 2] > 0.07]

    return bowl, mug


def find_bottle_and_box(cloud: NDArray, platform_1: bool=False) -> Tuple[NDArray, NDArray]:

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


def mug_tree_segmentation(cloud: NDArray, short_platform: bool=False,
                          tall_platform: bool=False, max_pc_size: Optional[int]=2000) -> Tuple[NDArray, NDArray]:

    cloud = center_workspace(cloud)
    cloud = mask_workspace(cloud)

    mug_pcd, tree_pcd = find_mug_and_tree(cloud, short_mug_platform=short_platform, tall_mug_plaform=tall_platform)
    if max_pc_size is not None:
        if len(mug_pcd) > max_pc_size:
            mug_pcd, _ = utils.farthest_point_sample(mug_pcd, max_pc_size)
        if len(tree_pcd) > max_pc_size:
            tree_pcd, _ = utils.farthest_point_sample(tree_pcd, max_pc_size)

    return mug_pcd, tree_pcd


def bowl_mug_segmentation(cloud: NDArray, max_pc_size: Optional[int]=2000, platform_1: bool=False) -> Tuple[NDArray, NDArray]:

    cloud = center_workspace(cloud)
    cloud = mask_workspace(cloud)

    bowl_pcd, mug_pcd = find_bowl_and_mug(cloud, platform_1=platform_1)
    if max_pc_size is not None:
        if len(bowl_pcd) > max_pc_size:
            bowl_pcd, _ = utils.farthest_point_sample(bowl_pcd, max_pc_size)
        if len(mug_pcd) > max_pc_size:
            mug_pcd, _ = utils.farthest_point_sample(mug_pcd, max_pc_size)

    return bowl_pcd, mug_pcd


def bottle_box_segmentation(cloud: NDArray, max_pc_size: Optional[int]=2000) -> Tuple[NDArray, NDArray]:

    cloud = center_workspace(cloud)
    cloud = mask_workspace(cloud)

    bottle_pcd, box_pcd = find_bottle_and_box(cloud)
    if max_pc_size is not None:
        if len(bottle_pcd) > max_pc_size:
            bottle_pcd, _ = utils.farthest_point_sample(bottle_pcd, max_pc_size)
        if len(box_pcd) > max_pc_size:
            box_pcd, _ = utils.farthest_point_sample(box_pcd, max_pc_size)

    return bottle_pcd, box_pcd


def warping(
    source_pcd: NDArray, target_pcd: NDArray,
    canon_source: utils.CanonObj, canon_target: utils.CanonObj,
    tf_proxy: Optional[TFProxy]=None,
    moveit_scene: Optional[MoveItScene]=None,
    source_save_decomposition: bool=False,
    target_save_decomposition: bool=False,
    add_source_to_planning_scene: bool=False,
    add_target_to_planning_scene: bool=False,
    rviz_pub: Optional[RVizPub]=None,
    ablate_no_warping: bool=False,
    any_rotation: bool=False,
    desk_center: Tuple[float, float, float]=constants.DESK_CENTER,
    grow_source_object: bool=False, grow_target_object: bool=False
    ) -> Tuple[NDArray, utils.ObjParam, NDArray, utils.ObjParam, utils.CanonObj, utils.CanonObj, NDArray, NDArray]:

    if ablate_no_warping:
        raise NotImplementedError()

    dc = np.array(desk_center)
    perception_start = time.time()

    if any_rotation:
        warp = object_warping.ObjectWarpingSE3Batch(
            canon_source, source_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=1000, object_size_reg=0.1, scaling=True, init_scale=1.)
        source_pcd_complete, _, source_param = object_warping.warp_to_pcd_se3(warp, n_angles=12, n_batches=15)
    else:
        warp = object_warping.ObjectWarpingSE2Batch(
            canon_source, source_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=1000, object_size_reg=0.1, scaling=True, init_scale=1.)
        source_pcd_complete, _, source_param = object_warping.warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

    warp = object_warping.ObjectWarpingSE2Batch(
        canon_target, target_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
        n_samples=1000, object_size_reg=0.1, scaling=True, init_scale=1.)
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
