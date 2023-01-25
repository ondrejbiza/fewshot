from typing import Optional, Tuple
import pickle
import trimesh
import numpy as np
from numpy.typing import NDArray
import moveit_commander

from online_isec.point_cloud_proxy import PointCloudProxy
from online_isec.tf_proxy import TFProxy
import online_isec.utils as isec_utils
import utils
import viz_utils


def mug_tree_perception(pc_proxy: PointCloudProxy, desk_center: NDArray, tf_proxy: Optional[TFProxy]=None,
                        moveit_scene: Optional[moveit_commander.PlanningSceneInterface]=None,
                        close_proxy: bool=True, max_pc_size: Optional[int]=2000,
                        canon_mug_path: str="data/ndf_mugs_pca_4_dim.npy",
                        canon_tree_path: str="data/real_tree_pc.pkl",
                        mug_save_decomposition: bool=True,
                        add_mug_to_planning_scene: bool=True,
                        add_tree_to_planning_scene: bool=True) -> Tuple[NDArray, NDArray, NDArray, NDArray]:

    cloud = pc_proxy.get_all()
    assert cloud is not None
    if close_proxy:
        pc_proxy.close()

    cloud = isec_utils.mask_workspace(cloud, (*pc_proxy.desk_center, pc_proxy.z_min))
    # viz_utils.o3d_visualize(utils.create_o3d_pointcloud(cloud))

    mug_pc, tree_pc = isec_utils.find_mug_and_tree(cloud)
    if max_pc_size is not None:
        if len(mug_pc) > max_pc_size:
            mug_pc, _ = utils.farthest_point_sample(mug_pc, max_pc_size)
        if len(tree_pc) > max_pc_size:
            tree_pc, _ = utils.farthest_point_sample(tree_pc, max_pc_size)

    # load canonical objects
    with open(canon_mug_path, "rb") as f:
        canon_mug = pickle.load(f)
    with open(canon_tree_path, "rb") as f:
        canon_tree = pickle.load(f)

    mug_pc_complete, _, mug_param = utils.planar_pose_warp_gd(canon_mug["pca"], canon_mug["canonical_obj"], mug_pc, object_size_reg=0.1, n_angles=12)
    tree_pc_complete, _, tree_param = utils.planar_pose_gd(canon_tree["canonical_obj"], tree_pc, n_angles=12)
    viz_utils.show_scene({0: mug_pc_complete, 1: tree_pc_complete}, background=np.concatenate([mug_pc, tree_pc]))

    if mug_save_decomposition:
        vertices = utils.canon_to_pc(canon_mug, mug_param)[:len(canon_mug["canonical_mesh_points"])]
        mesh = trimesh.base.Trimesh(vertices=vertices, faces=canon_mug["canonical_mesh_faces"])
        mesh.export("tmp.stl")
        utils.convex_decomposition(mesh, "tmp.obj")

    if add_mug_to_planning_scene:
        assert moveit_scene is not None and tf_proxy is not None, "Need moveit_scene and tf_proxy to add an object to the planning scene."
        pos, quat = utils.transform_to_pos_quat(isec_utils.desk_obj_param_to_base_link_T(mug_param[1], mug_param[2], desk_center, tf_proxy))
        isec_utils.load_obj_to_moveit_scene_2("tmp.stl", pos, quat, "mug", moveit_scene)

    if add_tree_to_planning_scene:
        assert moveit_scene is not None and tf_proxy is not None, "Need moveit_scene and tf_proxy to add an object to the planning scene."
        pos, quat = utils.transform_to_pos_quat(isec_utils.desk_obj_param_to_base_link_T(tree_param[0], tree_param[1], desk_center, tf_proxy))
        isec_utils.load_obj_to_moveit_scene_2("data/real_tree2.stl", pos, quat, "tree", moveit_scene)

    return mug_pc_complete, mug_param, tree_pc_complete, tree_param
