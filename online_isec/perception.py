from typing import Optional, Tuple, Dict, Any
import pickle
import trimesh
import numpy as np
from numpy.typing import NDArray
import moveit_commander
import pybullet as pb
from scipy.spatial.transform import Rotation

from online_isec.point_cloud_proxy_sync import PointCloudProxy
from online_isec.tf_proxy import TFProxy
import online_isec.utils as isec_utils
from online_isec.rviz_pub import RVizPub
from online_isec.moveit_scene import MoveItScene
import utils
import viz_utils
from pybullet_planning.pybullet_tools import utils as pu
from online_isec import constants
from online_isec.simulation import Simulation
from online_isec.ur5 import UR5


# TODO: finish typing
def mug_tree_perception(
    pc_proxy: PointCloudProxy, desk_center: NDArray, tf_proxy: Optional[TFProxy]=None,
    moveit_scene: Optional[MoveItScene]=None,
    close_proxy: bool=False, max_pc_size: Optional[int]=2000,
    canon_mug_path: str="data/230201_ndf_mugs_large_pca_8_dim.npy",
    canon_tree_path: str="data/real_tree_pc.pkl",
    mug_save_decomposition: bool=False,
    add_mug_to_planning_scene: bool=False,
    add_tree_to_planning_scene: bool=False,
    rviz_pub: Optional[RVizPub]=None,
    ablate_no_mug_warping: bool=False
    ) -> Tuple[NDArray, Tuple[NDArray, NDArray, NDArray], NDArray, Tuple[NDArray, NDArray], Dict[Any, Any], Dict[Any, Any]]:

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

    if ablate_no_mug_warping:
        mug_pc_complete, _, mug_param = utils.planar_pose_gd(canon_mug["canonical_obj"], mug_pc, n_angles=12)
        n_dimensions = canon_mug["pca"].n_components
        mug_param = (np.zeros(n_dimensions, dtype=np.float32), *mug_param)
    else:
        mug_pc_complete, _, mug_param = utils.planar_pose_warp_gd(canon_mug["pca"], canon_mug["canonical_obj"], mug_pc, object_size_reg=0.1, n_angles=12)

    tree_pc_complete, _, tree_param = utils.planar_pose_gd(canon_tree["canonical_obj"], tree_pc, n_angles=12)
    viz_utils.show_scene({0: mug_pc_complete, 1: tree_pc_complete}, background=np.concatenate([mug_pc, tree_pc]))

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

    return mug_pc_complete, mug_param, tree_pc_complete, tree_param, canon_mug, canon_tree


def perceive_mug_in_hand(pc_proxy: PointCloudProxy, sim: Simulation, ur5: UR5) -> NDArray[np.float32]:

    cloud = pc_proxy.get_all()
    assert cloud is not None

    # Add gripper to simulation.
    sim.remove_all_objects()

    g_pos, g_quat = ur5.get_end_effector_pose()

    # Bad attempt at aligning the gripper with the real-world one.
    delta = 0.005

    g_pos, g_quat = utils.move_hand_back((g_pos, g_quat), delta)
    T_g_to_b = utils.pos_quat_to_transform(g_pos, g_quat)
    T_f_to_g = ur5.tf_proxy.lookup_transform("flange", "tool0_controller")
    T_f_to_b = np.matmul(T_g_to_b, T_f_to_g)
    robotiq = sim.add_object("data/robotiq.urdf", *utils.transform_to_pos_quat(T_f_to_b))

    # Match the gripper joint.
    fract = ur5.gripper.get_open_fraction()
    # Normally we only move one joint, but joint mirroring doesn't work in pybullet.
    pu.set_joint_positions(robotiq, [0, 2, 4, 5, 6, 7], [fract, fract, fract, -fract, fract, -fract])

    # Cloud in workspace coordinates.
    cloud = isec_utils.mask_workspace(cloud, (*pc_proxy.desk_center, pc_proxy.z_min))
    mug_pc = isec_utils.find_held_mug(cloud)
    # Cloud in base coordinates.
    mug_pc = mug_pc + constants.DESK_CENTER

    # tool0_controller in base coordinates.
    g_pos, g_quat = ur5.get_end_effector_pose()
    # Mug in base coordinates.
    mug_pc = mug_pc - g_pos

    z_min, z_max = -0.08, 0.05
    mug_pc = mug_pc[np.logical_and(mug_pc[:, 2] >= z_min, mug_pc[:, 2] <= z_max)]

    # Identify points that might be part of the gripper.
    box_min = [-0.02, -0.02, 0.]
    box_max = [0.02, 0.02, 0.05]
    m1 = np.logical_and(mug_pc[:, 0] >= box_min[0], mug_pc[:, 0] <= box_max[0])
    m2 = np.logical_and(mug_pc[:, 1] >= box_min[1], mug_pc[:, 1] <= box_max[1])
    m3 = np.logical_and(mug_pc[:, 2] >= box_min[2], mug_pc[:, 2] <= box_max[2])
    mask1 = np.logical_and(m1, np.logical_and(m2, m3))
    indices = np.arange(len(mask1))[mask1]
    mask2 = np.ones(len(mask1), dtype=np.bool_)

    # Add a sphere for collision checking.
    sphere = sim.add_object("data/sphere_zero_radius.urdf", np.array([0., 0., 0.]), np.array([1., 0., 0., 0.]))

    # Filter out all points that are 5mm away from the gripper.
    delta = 0.005
    for i in indices:
        pos = mug_pc[i] + g_pos
        pu.set_pose(sphere, (pos, np.array([1., 0., 0., 0.])))

        pb.performCollisionDetection()
        cols = pb.getClosestPoints(sphere, robotiq, delta)
        if len(cols) > 0:
            mask2[i] = False

    # Hopefully clean mug PC.
    mug_pc = mug_pc[mask2]

    return mug_pc
