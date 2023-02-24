import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import open3d as o3d

from src import utils
from src.object_warping import ObjectWarpingSE2Batch, ObjectWarpingSE3Batch, warp_to_pcd_se2, warp_to_pcd_se3
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


def mug_tree_simple_perception(cloud: NDArray, max_pc_size: Optional[int]=2000) -> Tuple[NDArray, NDArray]:

    cloud = center_workspace(cloud)
    cloud = mask_workspace(cloud)

    mug_pc, tree_pc = find_mug_and_tree(cloud)
    if max_pc_size is not None:
        if len(mug_pc) > max_pc_size:
            mug_pc, _ = utils.farthest_point_sample(mug_pc, max_pc_size)
        if len(tree_pc) > max_pc_size:
            tree_pc, _ = utils.farthest_point_sample(tree_pc, max_pc_size)

    return mug_pc, tree_pc


def mug_tree_perception(
    cloud: NDArray, tf_proxy: Optional[TFProxy]=None,
    moveit_scene: Optional[MoveItScene]=None,
    max_pc_size: Optional[int]=2000,
    canon_mug_path: str="data/230213_ndf_mugs_scale_large_pca_8_dim_alp_0.01.pkl",
    canon_tree_path: str="data/230213_ndf_trees_scale_large_pca_8_dim_alp_2.pkl",
    mug_save_decomposition: bool=False,
    add_mug_to_planning_scene: bool=False,
    add_tree_to_planning_scene: bool=False,
    rviz_pub: Optional[RVizPub]=None,
    ablate_no_mug_warping: bool=False,
    any_rotation: bool=False,
    short_mug_platform: bool=False, tall_mug_platform: bool=False
    ) -> Tuple[NDArray, Tuple[NDArray, NDArray, NDArray], NDArray, Tuple[NDArray, NDArray], Dict[Any, Any], Dict[Any, Any], NDArray, NDArray]:

    cloud = center_workspace(cloud)
    cloud = mask_workspace(cloud)

    mug_pc, tree_pc = find_mug_and_tree(cloud, short_mug_platform=short_mug_platform, tall_mug_plaform=tall_mug_platform)
    if max_pc_size is not None:
        if len(mug_pc) > max_pc_size:
            mug_pc, _ = utils.farthest_point_sample(mug_pc, max_pc_size)
        if len(tree_pc) > max_pc_size:
            tree_pc, _ = utils.farthest_point_sample(tree_pc, max_pc_size)

    # load canonical objects
    canon_mug = utils.CanonObj.from_pickle(canon_mug_path)
    canon_tree = utils.CanonObj.from_pickle(canon_tree_path)

    perception_start = time.time()
    if ablate_no_mug_warping:
        mug_pc_complete, _, mug_param = utils.planar_pose_gd(canon_mug["canonical_obj"], mug_pc, n_angles=12)
        n_dimensions = canon_mug["pca"].n_components
        mug_param = (np.zeros(n_dimensions, dtype=np.float32), *mug_param)
    elif any_rotation:
        mug_pc_complete, _, mug_param = utils.pose_warp_gd_batch(canon_mug["pca"], canon_mug["canonical_obj"], mug_pc, object_size_reg=0.1, n_angles=60, n_batches=3)
    else:
        mug_pc_complete, _, mug_param = utils.planar_pose_warp_gd_batch(canon_mug["pca"], canon_mug["canonical_obj"], mug_pc, object_size_reg=0.1, n_angles=24)

    tree_pc_complete, _, tree_param = utils.planar_pose_gd(canon_tree["canonical_obj"], tree_pc, n_angles=12)
    viz_utils.show_scene({0: mug_pc_complete, 1: tree_pc_complete}, background=np.concatenate([mug_pc, tree_pc]))
    print("Inference time: {:.1f}s.".format(time.time() - perception_start))

    vertices = utils.canon_to_pc(canon_mug, mug_param)[:len(canon_mug["canonical_mesh_points"])]
    mesh = trimesh.base.Trimesh(vertices=vertices, faces=canon_mug["canonical_mesh_faces"])
    mesh.export("tmp.stl")
    if mug_save_decomposition:
        utils.convex_decomposition(mesh, "tmp.obj")

    if add_mug_to_planning_scene:
        assert moveit_scene is not None and tf_proxy is not None, "Need moveit_scene and tf_proxy to add an object to the planning scene."
        pos, quat = utils.transform_to_pos_quat(isec_utils.desk_obj_param_to_base_link_T(mug_param[1], mug_param[2], desk_center, tf_proxy))
        moveit_scene.add_object("tmp.stl", "mug", pos, quat)

    if add_tree_to_planning_scene:
        assert moveit_scene is not None and tf_proxy is not None, "Need moveit_scene and tf_proxy to add an object to the planning scene."
        pos, quat = utils.transform_to_pos_quat(isec_utils.desk_obj_param_to_base_link_T(tree_param[0], tree_param[1], desk_center, tf_proxy))
        moveit_scene.add_object("data/real_tree2.stl", "tree", pos, quat)

    if rviz_pub is not None:
        # send STL meshes as Marker messages to rviz
        assert tf_proxy is not None, "Need tf_proxy to calculate poses."

        pos, quat = utils.transform_to_pos_quat(isec_utils.desk_obj_param_to_base_link_T(mug_param[1], mug_param[2], desk_center, tf_proxy))
        rviz_pub.send_stl_message("tmp.stl", pos, quat)

        pos, quat = utils.transform_to_pos_quat(isec_utils.desk_obj_param_to_base_link_T(tree_param[0], tree_param[1], desk_center, tf_proxy))
        rviz_pub.send_stl_message("data/real_tree2.stl", pos, quat)

    return mug_pc_complete, mug_param, tree_pc_complete, tree_param, canon_mug, canon_tree, mug_pc, tree_pc
