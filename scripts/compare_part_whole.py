import os, os.path as osp
import sys
import random
import numpy as np
import pickle
from src import utils, viz_utils
import time
import signal
import torch
import argparse
import shutil
import threading
import json
import trimesh
import copy as cp
from scipy.spatial.transform import Rotation
from src.utils import CanonPart, CanonPartMetadata

import pybullet as p
import meshcat
import open3d as o3d

from airobot import log_info, log_warn, log_debug, log_critical, set_log_level
from airobot.utils import common
from airobot.utils.pb_util import create_pybullet_client
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet

import rndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from rndf_robot.utils import util, path_util

from rndf_robot.opt.optimizer import OccNetOptimizer
from rndf_robot.robot.multicam import MultiCams
from rndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from rndf_robot.share.globals import (
    bad_shapenet_mug_ids_list,
    bad_shapenet_bowls_ids_list,
    bad_shapenet_bottles_ids_list,
)
from rndf_robot.utils.pb2mc.pybullet_meshcat import PyBulletMeshcat
from rndf_robot.utils.eval_gen_utils import (
    constraint_obj_world,
    safeCollisionFilterPair,
    safeRemoveConstraint,
)

from rndf_robot.eval.relation_tools.multi_ndf import (
    infer_relation_intersection,
    create_target_descriptors,
)

from src.ndf_interface import NDFInterface, NDFPartInterface
from src.real_world import constants

def object_is_intersecting(obj1_id, obj2_id, obj1_link_index, obj2_link_index,
    max_force_threshold=.5):
    """
    During intersection, normal force is much higher than during contact.  If
    we use a threshold, we can determine if a teleport is in intersection or
    if it is placed correctly.

    Args:
        obj1_id (int): Id of first object to check
        obj2_id (int): Id of second object to check
        obj1_link_index (int): Link index for first obj, -1 for base object
        obj2_link_index (int): Link index for second obj, -1 for base object
        max_force_threshold (float, optional): Any max normal force above this
            is an intersection. Defaults to 5.

    Returns:
        bool: True for intersection, false otherwise.
    """
    # contact_points = p.getClosestPoints(bodyA=obj1_id, bodyB=obj2_id, distance=0.002,
    #     linkIndexA=obj1_link_index, linkIndexB=obj2_link_index)
    contact_points = p.getContactPoints(bodyA=obj1_id, bodyB=obj2_id,
        linkIndexA=obj1_link_index, linkIndexB=obj2_link_index)

    # Normal force should be idx 9
    max_normal_force = max([abs(point[9]) for point in contact_points] + [0])

    print('Max normal force: ', max_normal_force)
    return max_normal_force > max_force_threshold

def pb2mc_update(recorder, mc_vis, stop_event, run_event):
    iters = 0
    # while True:
    while not stop_event.is_set():
        run_event.wait()
        iters += 1
        recorder.add_keyframe()
        recorder.update_meshcat_current_state(mc_vis)
        time.sleep(1 / 230.0)


def check_segmentation_exists(pcl_id):
    root = "Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390"
    fn = os.path.join(root, pcl_id + ".txt")
    return os.path.exists(fn)


def pybullet_load_object(
    cfg,
    pc_master_dict,
    upright_orientation_dict,
    pb_client,
    table_id,
    table_base_id,
    new_parent_scale,
    demo_idx,
):
    poses = {}
    for pc in ["child", "parent"]:
        # get the object scales we will use
        scale_high, scale_low = pc_master_dict[pc]["scale_hl"]
        if pc == "parent":
            if new_parent_scale is None:
                scale_default = pc_master_dict[pc]["scale_default"]
            else:
                log_warn(
                    f"Setting new parent scale to: {new_parent_scale:.3f} to ensure parent is large enough for child"
                )
                scale_default = new_parent_scale
        else:
            scale_default = pc_master_dict[pc]["scale_default"]

        if args.rand_mesh_scale:
            mesh_scale = [np.random.random() * (scale_high - scale_low) + scale_low] * 3
        else:
            mesh_scale = [scale_default] * 3

        pc_master_dict[pc]["mesh_scale"] = mesh_scale

        object_class = pc_master_dict[pc]["class"]
        upright_orientation = upright_orientation_dict[object_class]

        # sample a pose to use for each object, depending on distribution of poses for this run
        load_pose_type = pc_master_dict[pc]["load_pose_type"]
        x_high, x_low = pc_master_dict[pc]["xhl"]
        y_high, y_low = pc_master_dict[pc]["yhl"]

        if args.sweep:
            print()
            # load_pose_type = "demo_pose"

        if load_pose_type == "any_pose":
            if object_class in ["bowl", "bottle"]:
                rp = np.random.rand(2) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rp[0], rp[1], 0]).tolist()
            else:
                rpy = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rpy[0], rpy[1], rpy[2]]).tolist()

            pos = [
                np.random.random() * (x_high - x_low) + x_low,
                np.random.random() * (y_high - y_low) + y_low,
                cfg.TABLE_Z,
            ]
            pose = pos + ori
            rand_yaw_T = util.rand_body_yaw_transform(
                pos, min_theta=-np.pi, max_theta=np.pi
            )
            pose_w_yaw = util.transform_pose(
                util.list2pose_stamped(pose), util.pose_from_matrix(rand_yaw_T)
            )
            pos, ori = (
                util.pose_stamped2list(pose_w_yaw)[:3],
                util.pose_stamped2list(pose_w_yaw)[3:],
            )
        else:
            if load_pose_type == "demo_pose":
                obj_start_pose_demo = pc_master_dict[pc]["demo_start_poses"][demo_idx]
                pos, ori = obj_start_pose_demo[:3], obj_start_pose_demo[3:]
            else:
                pos = [
                    np.random.random() * (x_high - x_low) + x_low,
                    np.random.random() * (y_high - y_low) + y_low,
                    cfg.TABLE_Z,
                ]
                pose = util.list2pose_stamped(pos + upright_orientation)
                rand_yaw_T = util.rand_body_yaw_transform(
                    pos, min_theta=-np.pi, max_theta=np.pi
                )
                pose_w_yaw = util.transform_pose(
                    pose, util.pose_from_matrix(rand_yaw_T)
                )
                pos, ori = (
                    util.pose_stamped2list(pose_w_yaw)[:3],
                    util.pose_stamped2list(pose_w_yaw)[3:],
                )
        poses[pc] = (pos, ori)
        # convert mesh with vhacd
        obj_obj_file, obj_obj_file_dec = (
            pc_master_dict[pc]["mesh_file"],
            pc_master_dict[pc]["mesh_file_dec"],
        )

        if not osp.exists(obj_obj_file_dec):
            p.vhacd(
                obj_obj_file,
                obj_obj_file_dec,
                "log.txt",
                concavity=0.0025,
                alpha=0.04,
                beta=0.05,
                gamma=0.00125,
                minVolumePerCH=0.0001,
                resolution=1000000,
                depth=20,
                planeDownsampling=4,
                convexhullDownsampling=4,
                pca=0,
                mode=0,
                convexhullApproximation=1,
            )

        # load the object into the simulator
        obj_id = pb_client.load_geom(
            "mesh",
            mass=0.01,
            mesh_scale=mesh_scale,
            visualfile=obj_obj_file_dec,
            collifile=obj_obj_file_dec,
            base_pos=pos,
            base_ori=ori,
        )

        # register the object with the meshcat visualizer
        # TODO: turned off for the sim thing
        # recorder.register_object(obj_id, obj_obj_file_dec, scaling=mesh_scale)

        # safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1, linkIndexB=rack_link_id, enableCollision=False)
        safeCollisionFilterPair(
            bodyUniqueIdA=obj_id,
            bodyUniqueIdB=table_id,
            linkIndexA=-1,
            linkIndexB=table_base_id,
            enableCollision=False,
        )
        p.changeDynamics(
            obj_id, -1, lateralFriction=0.5, linearDamping=5, angularDamping=5
        )

        # depending on the object/pose type, constrain the object to its world frame pose
        o_cid = None
        if (object_class in ["syn_rack_easy", "syn_rack_hard", "syn_rack_med"]) or (
            load_pose_type == "any_pose" and pc == "child"
        ):
            o_cid = constraint_obj_world(obj_id, pos, ori)
            pb_client.set_step_sim(False)
        pc_master_dict[pc]["o_cid"] = o_cid

        # safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        safeCollisionFilterPair(
            obj_id, table_id, -1, table_base_id, enableCollision=True
        )

        time.sleep(1.5)

        pc_master_dict[pc]["pb_obj_id"] = obj_id

        # poses[pc] = pb_client.get_body_state
    return poses


# For loading the part-segmented shapenet mugs
# That's currently hardcoded into the path
def load_segmented_pointcloud_from_txt(
    pcl_id,
    num_points=2048,
    root="Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390",
):
    fn = os.path.join(root, pcl_id + ".txt")
    cls = "Mug"
    data = np.loadtxt(fn).astype(np.float32)
    # if not self.normal_channel: #<-- ignore the normals
    point_set = data[:, 0:3]
    # else:
    # point_set = data[:, 0:6]
    seg_ids = data[:, -1].astype(np.int32)
    point_set[:, 0:3] = utils.center_pcl(point_set[:, 0:3])

    # fixed transform to align with the other mugs being used
    rotation = Rotation.from_euler("zyx", [0.0, np.pi / 2, 0.0]).as_quat()
    transform = utils.pos_quat_to_transform([0, 0, 0], rotation)
    point_set = utils.transform_pcd(point_set, transform)

    choice = np.random.choice(len(seg_ids), num_points, replace=True)
    return point_set, cls, seg_ids


def segment_mug_rack(rack_pts):
    # rack_pts = rack_mesh.vertices
    # Get the top points
    max_z = np.max(rack_pts[:, 2]) - 0.01
    max_pts = rack_pts[rack_pts[:, 2] > max_z]
    center = np.mean(max_pts, 0)

    rad = max(np.linalg.norm(max_pts - center, axis=-1)) * 5 / 4

    # rack_pts = utils.trimesh_create_verts_surface(rack_mesh, 2000)
    trunk = rack_pts[np.linalg.norm(rack_pts[:, :2] - center[:2], axis=-1) < rad]
    branch = rack_pts[np.linalg.norm(rack_pts[:, :2] - center[:2], axis=-1) > rad]
    seg_ids = np.zeros_like(rack_pts)
    seg_ids[np.linalg.norm(rack_pts[:, :2] - center[:2], axis=-1) > rad] = np.ones_like(
        seg_ids[np.linalg.norm(rack_pts[:, :2] - center[:2], axis=-1) > rad]
    )

    start_part_transforms = {
        "trunk": utils.pos_quat_to_transform(
            np.mean(trunk, 0), [0, 0, 0, 1]
        ),  # @ trans,
        "branch": utils.pos_quat_to_transform(np.mean(branch, 0), [0, 0, 0, 1]),
    }
    demo_parts = {
        "trunk": trunk,
        "branch": branch,
    }
    return demo_parts, seg_ids, start_part_transforms


def segment_mug(demo_pcl_id, trans):
    segmented_demo_mug, _, mug_seg_ids = load_segmented_pointcloud_from_txt(demo_pcl_id)

    # segmented_demo_mug = util.transform_pcd(segmented_demo_mug, trans)
    demo_cup = segmented_demo_mug[mug_seg_ids == 37]
    demo_handle = segmented_demo_mug[mug_seg_ids == 36]
    demo_mug = segmented_demo_mug

    # Hack to scale and repositions the parts, since the segmented pcls are normalized differently
    # from the raw shapenet data
    demo_cup, demo_cup_center = utils.center_pcl(demo_cup, return_centroid=True)
    demo_handle, demo_handle_center = utils.center_pcl(
        demo_handle, return_centroid=True
    )
    (
        demo_mug,
        demo_cup,
        demo_handle,
        demo_cup_center,
        demo_handle_center,
    ) = utils.scale_points_circle(
        [
            demo_mug,
            demo_cup,
            demo_handle,
            np.atleast_2d(demo_cup_center),
            np.atleast_2d(demo_handle_center),
        ],
        base_scale=0.13,
    )

    demo_cup = util.transform_pcd(
        demo_cup, utils.pos_quat_to_transform(demo_cup_center, [0, 0, 0, 1])
    )
    demo_cup = util.transform_pcd(demo_cup, trans)
    demo_handle = util.transform_pcd(
        demo_handle, utils.pos_quat_to_transform(demo_handle_center, [0, 0, 0, 1])
    )
    demo_handle = util.transform_pcd(demo_handle, trans)

    # viz_utils.show_pcds_plotly({'cup': demo_cup, 'handle':demo_handle, 'mug':pc_master_dict[pc]['demo_start_pcds'][i]})

    # TODO: Double check that this is necessary/these values actually change from the transform
    _, demo_cup_center = utils.center_pcl(demo_cup, return_centroid=True)
    _, demo_handle_center = utils.center_pcl(demo_handle, return_centroid=True)
    # _, demo_mug_center = utils.center_pcl(demo_mug, return_centroid=True)

    demo_parts = {
        "cup": demo_cup,
        "handle": demo_handle,
    }

    start_part_transforms = {
        "cup": utils.pos_quat_to_transform(demo_cup_center, [0, 0, 0, 1]),  # @ trans,
        "handle": utils.pos_quat_to_transform(demo_handle_center, [0, 0, 0, 1]),
    }
    adjusted_demo_parts = {"cup": demo_cup, "handle": demo_handle}
    return adjusted_demo_parts, -(mug_seg_ids - 37), start_part_transforms


def initialize_pointcloud_data(
    cfg,
    parent_class,
    child_class,
    demos,
    is_parent_shapenet_obj,
    is_child_shapenet_obj,
    parent_load_pose_type,
    child_load_pose_type,
):
    scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
    scale_default = cfg.MESH_SCALE_DEFAULT

    # cfg.OBJ_SAMPLE_Y_HIGH_LOW = [0.3, -0.3]
    cfg.OBJ_SAMPLE_Y_HIGH_LOW = [-0.35, 0.175]
    x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
    y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW

    pcl = ["parent", "child"]
    pc_master_dict = dict(parent={}, child={})
    pc_master_dict["parent"]["class"] = parent_class
    pc_master_dict["child"]["class"] = child_class

    valid_load_pose_types = ["any_pose", "demo_pose", "random_upright"]
    assert (
        parent_load_pose_type in valid_load_pose_types
    ), f'Invalid string value for args.parent_load_pose_type! Must be in {", ".join(valid_load_pose_types)}'
    assert (
        child_load_pose_type in valid_load_pose_types
    ), f'Invalid string value for args.child_load_pose_type! Must be in {", ".join(valid_load_pose_types)}'

    pc_master_dict["parent"]["load_pose_type"] = parent_load_pose_type
    pc_master_dict["child"]["load_pose_type"] = child_load_pose_type

    # # load in ids for objects that can be used for testing
    pc_master_dict["parent"]["test_ids"] = np.loadtxt(
        osp.join(path_util.get_rndf_share(), "%s_test_object_split.txt" % parent_class),
        dtype=str,
    ).tolist()

    pc_master_dict["child"]["test_ids"] = np.loadtxt(
        osp.join(path_util.get_rndf_share(), "%s_test_object_split.txt" % child_class),
        dtype=str,
    ).tolist()

    pc_master_dict["child"]["train_ids"] = np.loadtxt(
        osp.join(path_util.get_rndf_share(), "%s_train_object_split.txt" % child_class),
        dtype=str,
    ).tolist()

    # # process these to remove the file type
    pc_master_dict["parent"]["test_ids"] = [
        val.split(".")[0] for val in pc_master_dict["parent"]["test_ids"]
    ]
    pc_master_dict["child"]["test_ids"] = [
        val.split(".")[0] for val in pc_master_dict["child"]["test_ids"]
    ]

    pc_master_dict["child"]["train_ids"] = [
        val.split(".")[0] for val in pc_master_dict["child"]["train_ids"]
    ]

    log_info(f'Test ids (parent): {", ".join(pc_master_dict["parent"]["test_ids"])}')
    log_info(f'Test ids (child): {", ".join(pc_master_dict["child"]["test_ids"])}')

    for pc in pcl:
        object_class = pc_master_dict[pc]["class"]
        if object_class == "mug":
            avoid_ids = bad_shapenet_mug_ids_list + cfg.MUG.AVOID_SHAPENET_IDS
        elif object_class == "bowl":
            avoid_ids = bad_shapenet_bowls_ids_list + cfg.BOWL.AVOID_SHAPENET_IDS
        elif object_class == "bottle":
            avoid_ids = bad_shapenet_bottles_ids_list + cfg.BOTTLE.AVOID_SHAPENET_IDS
        else:
            avoid_ids = []

        pc_master_dict[pc]["avoid_ids"] = avoid_ids

    pc_master_dict["parent"]["xhl"] = [x_high, x_low]
    # pc_master_dict['parent']['yhl'] = [y_high, 0.075]
    pc_master_dict["parent"]["yhl"] = [0.2, 0.075]
    # pc_master_dict['parent']['yhl'] = [y_high, 0.05]

    pc_master_dict["child"]["xhl"] = [x_high, x_low]
    # pc_master_dict['child']['yhl'] = [-0.2, y_low]
    pc_master_dict["child"]["yhl"] = [-0.2, -0.3]
    # pc_master_dict['child']['yhl'] = [-0.075, y_low]
    # pc_master_dict['child']['yhl'] = [-0.05, y_low]

    # get the class specific ranges for scaling the objects
    for pc in pcl:
        if pc_master_dict[pc]["class"] == "mug":
            pc_master_dict[pc]["scale_hl"] = [0.35, 0.25]
            pc_master_dict[pc]["scale_default"] = 0.3
        if pc_master_dict[pc]["class"] == "bowl":
            pc_master_dict[pc]["scale_hl"] = [0.325, 0.15]
            pc_master_dict[pc]["scale_default"] = 0.3
        if pc_master_dict[pc]["class"] == "bottle":
            pc_master_dict[pc]["scale_hl"] = [0.35, 0.2]
            pc_master_dict[pc]["scale_default"] = 0.3
        if (
            pc_master_dict[pc]["class"] == "syn_rack_easy"
            or pc_master_dict[pc]["class"] == "syn_rack_med_1"
        ):
            # pc_master_dict[pc]['scale_hl'] = [1.1, 0.9]
            # pc_master_dict[pc]['scale_default'] = 1.0
            pc_master_dict[pc]["scale_hl"] = [0.35, 0.25]
            pc_master_dict[pc]["scale_default"] = 0.3
        if pc_master_dict[pc]["class"] == "syn_container":
            pc_master_dict[pc]["scale_hl"] = [1.1, 0.9]
            pc_master_dict[pc]["scale_default"] = 1.0

    # pc_master_dict['parent']['scale_hl'] = [0.45, 0.25]
    # pc_master_dict['parent']['scale_default'] = 0.3
    # pc_master_dict['child']['scale_hl'] = [1.1, 0.9]
    # pc_master_dict['child']['scale_default'] = 1.0

    # pc_master_dict['parent']['scale_hl'] = [1.1, 0.9]
    # pc_master_dict['parent']['scale_default'] = 1.0
    # pc_master_dict['child']['scale_hl'] = [0.45, 0.25]
    # pc_master_dict['child']['scale_default'] = 0.3

    # put the data in our pc master
    pc_master_dict["parent"]["demo_start_pcds"] = []
    pc_master_dict["parent"]["demo_final_pcds"] = []
    pc_master_dict["child"]["demo_start_pcds"] = []
    pc_master_dict["child"]["demo_final_pcds"] = []
    for pc in pcl:
        for idx, demo in enumerate(demos):
            s_pcd = demo["multi_obj_start_pcd"].item()[pc]
            f_pcd = demo["multi_obj_final_pcd"].item()[pc]

            pc_master_dict[pc]["demo_start_pcds"].append(s_pcd)
            pc_master_dict[pc]["demo_final_pcds"].append(f_pcd)

    # load data from demos in case we want to test on the shapes we trained on
    for pc in pcl:
        pc_master_dict[pc]["demo_ids"] = [
            dat["multi_object_ids"].item()[pc] for dat in demos
        ]
        print(pc_master_dict[pc]["demo_ids"])

        pc_master_dict[pc]["demo_start_poses"] = [
            dat["multi_obj_start_obj_pose"].item()[pc] for dat in demos
        ]
        pc_master_dict[pc]["demo_final_poses"] = [
            dat["multi_obj_final_obj_pose"].item()[pc] for dat in demos
        ]

        # For recording the pose of each demonstration part
        pc_master_dict[pc]["demo_start_part_pcds"] = []
        pc_master_dict[pc]["demo_start_part_poses"] = []
        pc_master_dict[pc]["demo_final_part_poses"] = []
    return pc_master_dict


def load_mesh_info():
    mesh_data_dirs = {
        "mug": "mug_centered_obj_normalized",
        "bottle": "bottle_centered_obj_normalized",
        "bowl": "bowl_centered_obj_normalized",
        "syn_rack_easy": "syn_racks_easy_obj",
        "syn_rack_med_1": "syn_racks_med_1",
        "syn_container": "box_containers_unnormalized",
    }
    mesh_data_dirs = {
        k: osp.join(path_util.get_rndf_obj_descriptions(), v)
        for k, v in mesh_data_dirs.items()
    }

    bad_ids = {
        "syn_rack_easy": [],
        "bowl": bad_shapenet_bowls_ids_list,
        "mug": bad_shapenet_mug_ids_list,
        "bottle": bad_shapenet_bottles_ids_list,
        "syn_rack_med_1": [],
        "syn_container": [],
    }

    upright_orientation_dict = {
        "mug": common.euler2quat([np.pi / 2, 0, 0]).tolist(),
        "bottle": common.euler2quat([np.pi / 2, 0, 0]).tolist(),
        "bowl": common.euler2quat([np.pi / 2, 0, 0]).tolist(),
        "syn_rack_easy": common.euler2quat([0, 0, 0]).tolist(),
        "syn_rack_med_1": common.euler2quat([0, 0, 0]).tolist(),
        "syn_container": common.euler2quat([0, 0, 0]).tolist(),
    }

    mesh_names = {}
    for k, v in mesh_data_dirs.items():
        # get train samples
        objects_raw = os.listdir(v)
        objects_filtered = [
            fn
            for fn in objects_raw
            if (fn.split("/")[-1] not in bad_ids[k] and "_dec" not in fn)
        ]

        # objects_filtered = objects_raw
        total_filtered = len(objects_filtered)
        train_n = int(total_filtered * 0.9)
        test_n = total_filtered - train_n

        train_objects = sorted(objects_filtered)[:train_n]
        test_objects = sorted(objects_filtered)[train_n:]

        log_info("\n\n\nTest objects: ")
        log_info(test_objects)
        # log_info('\n\n\n')

        mesh_names[k] = objects_filtered

    return mesh_names, mesh_data_dirs, upright_orientation_dict


def main(args):
    #####################################################################################
    # set up all generic experiment info
    pcl = ["child", "parent"]

    # Canon source path disabled to be replaced to the paths with the object parts
    if args.exp == "bowl_on_mug_upright_pose_new":
        source_part_names = ["whole_bowl"]
        target_part_names = ["cup", "handle"]
        source_whole_name = ["whole_bowl"]
        target_whole_name = ["whole_mug"]

        canon_source_scale = constants.NDF_BOWLS_INIT_SCALE
        canon_target_scale = constants.NDF_MUGS_INIT_SCALE

    elif args.exp == "mug_on_rack_upright_pose_new":
        source_part_names = ["cup", "handle"]
        target_part_names = ["trunk", "branch"]
        source_whole_name = ["whole_mug"]
        target_whole_name = ["whole_syn_rack_easy"]

        canon_source_scale = constants.NDF_MUGS_INIT_SCALE
        canon_target_scale = constants.NDF_TREES_INIT_SCALE

    elif args.exp == "mug_on_rack_sweep":
        source_part_names = ["cup", "handle"]
        target_part_names = ["trunk", "branch"]
        source_whole_name = ["whole_mug"]
        target_whole_name = ["whole_syn_rack_easy"]

        canon_source_scale = constants.NDF_MUGS_INIT_SCALE
        canon_target_scale = constants.NDF_TREES_INIT_SCALE

    elif args.exp == "mug_on_rack_med_1_pose_new":
        source_part_names = ["cup", "handle"]
        target_part_names = ["syn_rack_med_1_trunk", "syn_rack_med_1_branch"]
        source_whole_name = ["whole_mug"]
        target_whole_name = ["whole_syn_rack_med_1"]

        canon_source_scale = constants.NDF_MUGS_INIT_SCALE
        canon_target_scale = constants.NDF_TREES_INIT_SCALE

    elif args.exp == "bottle_in_container_upright_pose_new":
        source_part_names = ["?"]
        target_part_names = ["?"]
        source_whole_name = ["?"]
        target_whole_name = ["?"]

        canon_source_scale = constants.NDF_BOTTLES_INIT_SCALE
        canon_target_scale = constants.BOXES_INIT_SCALE

    else:
        raise ValueError("Unknown experiment.")

    # Setting up results recording
    if args.debug:
        set_log_level("debug")
    else:
        set_log_level("info")

    signal.signal(signal.SIGINT, util.signal_handler)

    demo_path = osp.join(path_util.get_rndf_data(), "relation_demos", args.rel_demo_exp)
    demo_files = [fn for fn in sorted(os.listdir(demo_path)) if fn.endswith(".npz")]
    demos = []
    for f in demo_files:
        demo = np.load(demo_path + "/" + f, mmap_mode="r", allow_pickle=True)
        demos.append(demo)

    # Naming the experiment save file
    import time

    timestr = time.strftime("%Y%m%d-%H%M%S")
    expstr = f"rel_affected_exp--{args.exp}_demo-exp--{args.rel_demo_exp}_{timestr}"
    seedstr = "seed--" + str(args.seed)
    experiment_name = "_".join([expstr, seedstr])

    eval_save_dir_root = osp.join(
        path_util.get_rndf_eval_data(), args.eval_data_dir, experiment_name
    )
    eval_save_dir = eval_save_dir_root

    util.safe_makedirs(eval_save_dir_root)
    util.safe_makedirs(eval_save_dir)

    # Set up visualization and recording
    zmq_url = "tcp://127.0.0.1:6000"
    log_warn(f"Starting meshcat at zmq_url: {zmq_url}")
    mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
    mc_vis["scene"].delete()

    pb_client = create_pybullet_client(
        gui=False, opengl_render=True, realtime=False, server=args.pybullet_server
    )  # args.pybullet_viz
    recorder = PyBulletMeshcat(pb_client=pb_client)
    recorder.clear()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # general experiment + environment setup/scene generation configs
    cfg = get_eval_cfg_defaults()
    config_fname = osp.join(path_util.get_rndf_config(), "eval_cfgs", args.config)
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        log_info(f"Config file {config_fname} does not exist, using defaults")
    # cfg.freeze()

    eval_teleport_imgs_dir = osp.join(eval_save_dir, "teleport_imgs")
    util.safe_makedirs(eval_teleport_imgs_dir)

    save_dir = osp.join(path_util.get_rndf_eval_data(), "multi_class", args.exp)
    util.safe_makedirs(save_dir)

    #####################################################################################
    # load in the parent/child model and the cameras

    cams = MultiCams(cfg.CAMERA, pb_client, n_cams=cfg.N_CAMERAS)
    cam_info = {}
    cam_info["pose_world"] = []
    for cam in cams.cams:
        cam_info["pose_world"].append(util.pose_from_matrix(cam.cam_ext_mat))

    table_z = cfg.TABLE_Z

    #####################################################################################
    # load all the multi class mesh info

    mesh_names, mesh_data_dirs, upright_orientation_dict = load_mesh_info()

    obj_classes = list(mesh_names.keys())

    #####################################################################################
    # load all the parent/child info
    parent_class = args.parent_class
    child_class = args.child_class
    is_parent_shapenet_obj = args.is_parent_shapenet_obj
    is_child_shapenet_obj = args.is_child_shapenet_obj

    pc_master_dict = initialize_pointcloud_data(
        cfg,
        parent_class,
        child_class,
        demos,
        is_parent_shapenet_obj,
        is_child_shapenet_obj,
        args.parent_load_pose_type,
        args.child_load_pose_type,
    )

    #####################################################################################

    # Replacing the demo pointcloud with the segmented version of the same object
    for pc in ["child", "parent"]:
        for i in range(len(pc_master_dict[pc]["demo_start_pcds"])):
            demo_pcl_id = pc_master_dict[pc]["demo_ids"][i]
            trans = utils.pos_quat_to_transform(
                pc_master_dict[pc]["demo_start_poses"][i][:3],
                pc_master_dict[pc]["demo_start_poses"][i][3:],
            )

            if pc_master_dict[pc]["class"] == "mug":
                try:
                    adjusted_demo_parts, _, start_part_transforms = segment_mug(
                        demo_pcl_id, trans
                    )
                except FileNotFoundError as e:
                    print(
                        f"SKIPPING DEMO #{i}, ID: {demo_pcl_id}. NO SEGMENTED FORM FOUND"
                    )
                    pc_master_dict[pc]["demo_start_part_pcds"].append(None)
                    pc_master_dict[pc]["demo_start_part_poses"].append(None)
                    pc_master_dict[pc]["demo_final_part_poses"].append(None)
                    continue
            elif (
                pc_master_dict[pc]["class"] == "syn_rack_easy"
                or pc_master_dict[pc]["class"] == "syn_rack_med_1"
            ):
                adjusted_demo_parts, _, start_part_transforms = segment_mug_rack(
                    pc_master_dict[pc]["demo_start_pcds"][i]
                )
            else:
                adjusted_demo_parts = {
                    source_part_names[0]: pc_master_dict[pc]["demo_start_pcds"][i]
                }
                start_part_transforms = {source_part_names[0]: trans}

            pc_master_dict[pc]["demo_start_part_pcds"].append(adjusted_demo_parts)
            pc_master_dict[pc]["demo_start_part_poses"].append(start_part_transforms)

    #####################################################################################
    # prepare the simuation environment

    table_urdf_fname = osp.join(
        path_util.get_rndf_descriptions(), "hanging/table/table_manual.urdf"
    )
    # table_urdf_fname = osp.join(path_util.get_rndf_descriptions(), 'hanging/table/table_rack_manual.urdf')
    # table_urdf_fname = osp.join(path_util.get_rndf_descriptions(), 'hanging/table/table_rack.urdf')
    table_id = pb_client.load_urdf(
        table_urdf_fname, cfg.TABLE_POS, cfg.TABLE_ORI, scaling=1.0
    )
    recorder.register_object(table_id, table_urdf_fname)

    rec_stop_event = threading.Event()
    rec_run_event = threading.Event()
    rec_th = threading.Thread(
        target=pb2mc_update, args=(recorder, mc_vis, rec_stop_event, rec_run_event)
    )  # , mc_vis))
    rec_th.daemon = True
    rec_th.start()

    pause_mc_thread = (
        lambda pause_bool: rec_run_event.clear() if pause_bool else rec_run_event.set()
    )
    pause_mc_thread(False)

    table_base_id = 0
    rack_link_id = 0

    eval_imgs_dir = osp.join(eval_save_dir, "eval_imgs")
    util.safe_makedirs(eval_imgs_dir)
    eval_cam = RGBDCameraPybullet(cams._camera_cfgs(), pb_client)
    eval_cam.setup_camera(
        focus_pt=[0.4, 0.0, table_z], dist=0.9, yaw=45, pitch=-25, roll=0
    )

    #####################################################################################
    # dump full experiment configs in eval folder

    full_cfg_dict = {}
    for k, v in args.__dict__.items():
        full_cfg_dict[k] = v
    for k, v in util.cn2dict(cfg).items():
        full_cfg_dict[k] = v
    full_cfg_fname = osp.join(eval_save_dir, "full_exp_cfg.txt")
    json.dump(
        full_cfg_dict,
        open(full_cfg_fname, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )

    #####################################################################################

    canon_source_parts_paths = {
        part: f"part_based_warp_models/{part}_{args.canon_source_file_stamp}"
        for part in source_part_names
    }
    canon_target_parts_paths = {
        part: f"part_based_warp_models/{part}_{args.canon_target_file_stamp}"
        for part in target_part_names
    }

    canon_source_parts = {
        part: pickle.load(open(canon_source_parts_paths[part], "rb"))
        for part in source_part_names
    }
    canon_target_parts = {
        part: pickle.load(open(canon_target_parts_paths[part], "rb"))
        for part in target_part_names
    }

    canon_source_training_ids = canon_source_parts[
        source_part_names[0]
    ].metadata.training_ids + [
        canon_source_parts[source_part_names[0]].metadata.canonical_id
    ]
    canon_target_training_ids = canon_target_parts[
        target_part_names[0]
    ].metadata.training_ids + [
        canon_target_parts[target_part_names[0]].metadata.canonical_id
    ]

    # Set up parts-based interface
    part_interface = NDFPartInterface(
        canon_source_parts_paths=canon_source_parts_paths,
        canon_target_parts_paths=canon_target_parts_paths,
        canon_source_scale=canon_source_scale,
        canon_target_scale=canon_target_scale,
        source_part_names=source_part_names,
        target_part_names=target_part_names,
        ablate_no_warp=args.ablate_no_warp,
        ablate_no_scale=args.ablate_no_scale,
        ablate_no_pose_training=args.ablate_no_pose_training,
        ablate_no_size_reg=args.ablate_no_size_reg,
    )

    canon_source_whole_path = {
        part: f"part_based_warp_models/{part}_{args.canon_source_file_stamp}"
        for part in source_whole_name
    }
    canon_target_whole_path = {
        part: f"part_based_warp_models/{part}_{args.canon_target_file_stamp}"
        for part in target_whole_name
    }

    # Set up whole object interface
    whole_interface = NDFInterface(
        canon_source_path=canon_source_whole_path,
        canon_target_path=canon_target_whole_path,
        canon_source_scale=canon_source_scale,
        canon_target_scale=canon_target_scale,
        source_whole_name=source_whole_name,
        target_whole_name=target_whole_name,
        ablate_no_warp=args.ablate_no_warp,
        ablate_no_scale=args.ablate_no_scale,
        ablate_no_pose_training=args.ablate_no_pose_training,
        ablate_no_size_reg=args.ablate_no_size_reg,
    )
    import time

    timestr = time.strftime("%Y%m%d-%H%M%S")

    for interface, exp_type in zip(
        [ part_interface, whole_interface], [ "parts_based", "whole"]
    ):
        #####################################################################################
        # set the demo info
        max_demos = len(demos)
        n_demos = max_demos if args.n_demos == 0 else args.n_demos

        # Picks the demo with the best reconstruction if demo_selection is True
        demo_idx = args.demo_idx
        if args.demo_selection:
            costs, programs = [], []
            for i in range(n_demos):
                try:
                    cost, program = interface.set_demo_info(
                        pc_master_dict, demo_idx=i, calculate_cost=True, show=True
                    )
                    costs.append(cost)
                    programs.append(program)
                except TypeError as e:
                    print(e)
                    print(f"SKIPPING {i}")
                    continue
            print("costs:", costs)
            demo_idx = int(np.argmin(costs))

        demo_cost, demo_program = interface.set_demo_info(
            pc_master_dict, demo_idx=demo_idx, calculate_cost=True
        )

        #####################################################################################
        # start experiment: sample parent and child object on each iteration and infer the relation
        place_success_list = []

        # folder for experiment results
        experiment_folder = "./experiment_results/"
        experiment_name = f"{exp_type}_{args.exp}_"

        experiment_path = experiment_folder + experiment_name + timestr + "/"

        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

        if args.sweep:
            if args.exp == "mug_on_rack_sweep":
                child_objects = (
                    pc_master_dict["child"]["test_ids"]
                    + pc_master_dict["child"]["train_ids"]
                )
                child_objects = [
                    x for x in child_objects if x not in canon_source_training_ids
                ]
                child_objects = [
                    x for x in child_objects if check_segmentation_exists(x)
                ]
                args.num_iterations = len(child_objects)

        for iteration in range(args.start_iteration, args.num_iterations):
            #####################################################################################
            # set up the trial

            # to skip the ones that don't have segmentations
            if args.test_on_train:
                while True:
                    demo_idx = np.random.randint(len(demos))
                    demo = demos[demo_idx]
                    if pc_master_dict[pc]["demo_start_part_pcds"][demo_idx] is not None:
                        break

                parent_id = pc_master_dict["parent"]["demo_ids"][demo_idx]
                while True:
                    child_id = pc_master_dict["child"]["demo_ids"][demo_idx]
                    if check_segmentation_exists(child_id):
                        break
            else:
                # parent_id = random.sample(pc_master_dict['parent']['test_ids'], 1)[0]
                # while True:
                #     child_id = random.sample(pc_master_dict['child']['test_ids'], 1)[0]
                #     if check_segmentation_exists(child_id):
                #         break
                if args.sweep:
                    if args.exp == "mug_on_rack_sweep":
                        child_id = child_objects[iteration]
                        # parent_id = pc_master_dict["parent"][demo]
                        # parent_objects = (
                        #     pc_master_dict["parent"]["test_ids"]
                        #     + pc_master_dict["parent"]["train_ids"]
                        # ) - canon_target_training_ids
                        # child_id = random.sample(child_objects, 1)[0]
                        parent_id = pc_master_dict["parent"]["demo_ids"][7]
                else:
                    if (
                        args.is_child_shapenet_obj
                        and pc_master_dict["child"]["class"] != "bowl"
                    ):
                        while True:
                            child_id = random.sample(
                                pc_master_dict["child"]["test_ids"], 1
                            )[0]
                            if check_segmentation_exists(child_id):
                                break
                    else:
                        child_id = random.sample(
                            pc_master_dict["child"]["test_ids"], 1
                        )[0]

                    if (
                        args.is_parent_shapenet_obj
                        and pc_master_dict["parent"]["class"] != "bowl"
                    ):
                        while True:
                            parent_id = random.sample(
                                pc_master_dict["parent"]["test_ids"], 1
                            )[0]
                            if check_segmentation_exists(parent_id):
                                break
                    else:
                        parent_id = random.sample(
                            pc_master_dict["parent"]["test_ids"], 1
                        )[0]

            if "_dec" in parent_id:
                parent_id = parent_id.replace("_dec", "")
            if "_dec" in child_id:
                child_id = child_id.replace("_dec", "")

            id_str = f"Parent ID: {parent_id}, Child ID: {child_id}"
            experiment_id = experiment_path + f"parent_{parent_id}_child_{child_id}"
            log_info(id_str)

            # make folder for saving this trial
            eval_iter_dir = osp.join(eval_save_dir, f"trial_{iteration}")
            util.safe_makedirs(eval_iter_dir)

            #####################################################################################
            # load parent/child objects into the scene -- mesh file, pose, and pybullet object id

            if is_parent_shapenet_obj:
                parent_obj_file = osp.join(
                    mesh_data_dirs[parent_class],
                    parent_id,
                    "models/model_normalized.obj",
                )
                parent_obj_file_dec = parent_obj_file.split(".obj")[0] + "_dec.obj"
            else:
                parent_obj_file = osp.join(
                    mesh_data_dirs[parent_class], parent_id + ".obj"
                )
                parent_obj_file_dec = parent_obj_file.split(".obj")[0] + "_dec.obj"

            if is_child_shapenet_obj:
                child_obj_file = osp.join(
                    mesh_data_dirs[child_class], child_id, "models/model_normalized.obj"
                )
                child_obj_file_dec = child_obj_file.split(".obj")[0] + "_dec.obj"
            else:
                child_obj_file = osp.join(
                    mesh_data_dirs[child_class], child_id + ".obj"
                )
                child_obj_file_dec = child_obj_file.split(".obj")[0] + "_dec.obj"

            new_parent_scale = None

            for pc in ["child", "parent"]:
                # get the mesh files we will use
                pc_master_dict[pc]["mesh_file"] = (
                    parent_obj_file if pc == "parent" else child_obj_file
                )
                pc_master_dict[pc]["mesh_file_dec"] = (
                    parent_obj_file_dec if pc == "parent" else child_obj_file_dec
                )

            poses = pybullet_load_object(
                cfg,
                pc_master_dict,
                upright_orientation_dict,
                pb_client,
                table_id,
                table_base_id,
                new_parent_scale,
                demo_idx,
            )

            # get object point cloud
            depth_imgs = []
            seg_idxs = []
            obj_pcd_pts = []

            pc_obs_info = {}
            pc_obs_info["pcd"] = {}
            pc_obs_info["pcd_pts"] = {}
            pc_obs_info["pcd_pts"]["parent"] = []
            pc_obs_info["pcd_pts"]["child"] = []

            for i, cam in enumerate(cams.cams):
                # get image and raw point cloud
                rgb, depth, seg = cam.get_images(
                    get_rgb=True, get_depth=True, get_seg=True
                )
                pts_raw, _ = cam.get_pcd(
                    in_world=True,
                    rgb_image=rgb,
                    depth_image=depth,
                    depth_min=0.0,
                    depth_max=np.inf,
                )

                # flatten and find corresponding pixels in segmentation mask
                flat_seg = seg.flatten()
                flat_depth = depth.flatten()

                for pc in pcl:
                    obj_id = pc_master_dict[pc]["pb_obj_id"]
                    obj_inds = np.where(flat_seg == obj_id)
                    seg_depth = flat_depth[obj_inds[0]]

                    obj_pts = pts_raw[obj_inds[0], :]
                    # obj_pcd_pts.append(util.crop_pcd(obj_pts))
                    pc_obs_info["pcd_pts"][pc].append(util.crop_pcd(obj_pts))

                depth_imgs.append(seg_depth)
                seg_idxs.append(obj_inds)

            # merge point clouds from different views, and filter weird artifacts away from the object
            for pc, obj_pcd_pts in pc_obs_info["pcd_pts"].items():
                target_obj_pcd_obs = np.concatenate(
                    obj_pcd_pts, axis=0
                )  # object shape point cloud
                target_pts_mean = np.mean(target_obj_pcd_obs, axis=0)
                inliers = np.where(
                    np.linalg.norm(target_obj_pcd_obs - target_pts_mean, 2, 1) < 0.2
                )[0]
                target_obj_pcd_obs = target_obj_pcd_obs[inliers]

                pc_obs_info["pcd"][pc] = target_obj_pcd_obs

            parent_pcd = pc_obs_info["pcd"]["parent"]
            child_pcd = pc_obs_info["pcd"]["child"]

            # TODO: increase generality here
            print(exp_type)
            # print(poses)

            parent_obj_id = pc_master_dict["parent"]["pb_obj_id"]
            child_obj_id = pc_master_dict["child"]["pb_obj_id"]
            actual_parent_pose = np.concatenate(
                pb_client.get_body_state(parent_obj_id)[:2]
            ).tolist()
            actual_child_pose = np.concatenate(
                pb_client.get_body_state(child_obj_id)[:2]
            ).tolist()

            if pc_master_dict["child"]["class"] == "mug":
                child_parts, child_labels, start_part_transforms = segment_mug(
                    child_id,
                    utils.pos_quat_to_transform(
                        actual_child_pose[:3], actual_child_pose[3:]
                    ),
                )
            else:
                child_parts = {"whole_bowl": child_pcd}

            if (
                pc_master_dict["parent"]["class"] == "syn_rack_easy"
                or pc_master_dict["parent"]["class"] == "syn_rack_med_1"
            ):
                parent_parts, _, start_parent_transforms = segment_mug_rack(parent_pcd)
            else:
                parent_parts, _, start_parent_transforms = segment_mug(
                    parent_id,
                    utils.pos_quat_to_transform(
                        actual_parent_pose[:3], actual_parent_pose[3:]
                    ),
                )  # segment_mug_rack(parent_pcd)

            parent_obj_id = pc_master_dict["parent"]["pb_obj_id"]
            actual_parent_pose = np.concatenate(
                pb_client.get_body_state(parent_obj_id)[:2]
            ).tolist()
            print(actual_parent_pose)

            # if args.is_parent_shapenet_obj:

            log_info(f"[INTERSECTION], Loading model weights for multi NDF inference")

            se3 = False
            if args.child_load_pose_type == "any_pose":
                se3 = True

            pause_mc_thread(True)
            if exp_type == "parts_based":
                relative_trans = interface.infer_relpose(
                    child_parts,
                    parent_parts,
                    demo_program,
                    se3=se3,
                    experiment_id=experiment_id,
                )
            else:
                relative_trans = interface.infer_relpose(
                    child_pcd, parent_pcd, None, se3=se3, experiment_id=experiment_id
                )
            pause_mc_thread(False)
            time.sleep(1.0)

            # apply the inferred transformation by updating the pose of the child object
            parent_obj_id = pc_master_dict["parent"]["pb_obj_id"]
            child_obj_id = pc_master_dict["child"]["pb_obj_id"]
            start_child_pose = np.concatenate(
                pb_client.get_body_state(child_obj_id)[:2]
            ).tolist()
            start_child_pose_mat = util.matrix_from_pose(
                util.list2pose_stamped(start_child_pose)
            )
            final_child_pose_mat = np.matmul(relative_trans, start_child_pose_mat)

            start_parent_pose = np.concatenate(
                pb_client.get_body_state(parent_obj_id)[:2]
            ).tolist()
            start_parent_pose_mat = util.matrix_from_pose(
                util.list2pose_stamped(start_parent_pose)
            )
            upright_orientation = upright_orientation_dict[
                pc_master_dict["parent"]["class"]
            ]
            upright_parent_ori_mat = common.quat2rot(upright_orientation)

            pb_client.set_step_sim(True)
            if pc_master_dict["parent"]["load_pose_type"] == "any_pose":
                # TODO: ???????
                # get the relative transformation to make it upright
                upright_parent_pose_mat = copy.deepcopy(start_parent_pose_mat)
                upright_parent_pose_mat[:-1, :-1] = upright_parent_ori_mat
                relative_upright_pose_mat = np.matmul(
                    upright_parent_pose_mat, np.linalg.inv(start_parent_pose_mat)
                )

                upright_parent_pos, upright_parent_ori = start_parent_pose[
                    :3
                ], common.rot2quat(upright_parent_ori_mat)
                pb_client.reset_body(
                    parent_obj_id, upright_parent_pos, upright_parent_ori
                )

                final_child_pose_mat = np.matmul(
                    relative_upright_pose_mat, final_child_pose_mat
                )

            final_child_pose_list = util.pose_stamped2list(
                util.pose_from_matrix(final_child_pose_mat)
            )
            final_child_pos, final_child_ori = (
                final_child_pose_list[:3],
                final_child_pose_list[3:],
            )

            print(f"Start parent: {start_parent_pose}")
            print(f"from master dict: {poses['parent'][0]}")

            print(f"Start child: {start_child_pose}")
            print(f"from master dict: {poses['child'][0]}")

            # apply computed final pose by resetting the state
            pb_client.reset_body(child_obj_id, final_child_pos, final_child_ori)
            if pc_master_dict["parent"]["class"] not in [
                "syn_rack_easy",
                "syn_rack_med",
            ]:
                safeRemoveConstraint(pc_master_dict["parent"]["o_cid"])
            if pc_master_dict["child"]["class"] not in [
                "syn_rack_easy",
                "syn_rack_med",
            ]:
                safeRemoveConstraint(pc_master_dict["child"]["o_cid"])

            final_child_pcd = util.transform_pcd(
                pc_obs_info["pcd"]["child"], relative_trans
            )

            fig = viz_utils.show_pcds_plotly(
                {
                    "start_child": child_pcd,
                    "final_child": final_child_pcd,
                    "parent_pcd": parent_pcd,
                }
            )
            fig.show()
            # input("continue2?")

            with recorder.meshcat_scene_lock:
                util.meshcat_pcd_show(
                    mc_vis,
                    final_child_pcd,
                    color=[255, 0, 255],
                    name="scene/final_child_pcd",
                )

            # safeCollisionFilterPair(pc_master_dict['child']['pb_obj_id'], table_id, -1, -1, enableCollision=False)
            safeCollisionFilterPair(
                pc_master_dict["child"]["pb_obj_id"],
                table_id,
                -1,
                table_base_id,
                enableCollision=False,
            )

            time.sleep(3.0)

            pb_client.stepSimulation()
            intersecting = object_is_intersecting(child_obj_id, parent_obj_id, -1, -1)

            # turn on the physics and let things settle to evaluate success/failure
            pb_client.set_step_sim(False)

            # evaluation criteria
            time.sleep(2.0)
            success_crit_dict = {}
            kvs = {}

            obj_surf_contacts = p.getContactPoints(
                pc_master_dict["child"]["pb_obj_id"],
                pc_master_dict["parent"]["pb_obj_id"],
                -1,
                -1,
            )
            touching_surf = len(obj_surf_contacts) > 0
            success_crit_dict["touching_surf"] = touching_surf
            if parent_class == "syn_container" and child_class == "bottle":
                bottle_final_pose = np.concatenate(
                    p.getBasePositionAndOrientation(
                        pc_master_dict["child"]["pb_obj_id"]
                    )[:2]
                ).tolist()

                # get the y-axis in the body frame
                bottle_body_y = common.quat2rot(bottle_final_pose[3:])[:, 1]
                bottle_body_y = bottle_body_y / np.linalg.norm(bottle_body_y)

                # get the angle deviation from the vertical
                angle_from_upright = util.angle_from_3d_vectors(
                    bottle_body_y, np.array([0, 0, 1])
                )
                bottle_upright = angle_from_upright < args.upright_ori_diff_thresh
                success_crit_dict["bottle_upright"] = bottle_upright

            # take an image to make sure it looks good (post-process)
            eval_rgb = eval_cam.get_images(get_rgb=True)[0]
            eval_img_fname = osp.join(eval_imgs_dir, f"{iteration}.png")
            util.np2img(eval_rgb.astype(np.uint8), eval_img_fname)

            ##########################################################################
            ##upside down check for too much inter-penetration
            # pb_client.set_step_sim(True)

            # # remove constraints, if there are any
            # safeRemoveConstraint(pc_master_dict["parent"]["o_cid"])
            # safeRemoveConstraint(pc_master_dict["child"]["o_cid"])

            # # first, reset everything
            # pb_client.reset_body(
            #     parent_obj_id, start_parent_pose[:3], start_parent_pose[3:]
            # )
            # pb_client.reset_body(
            #     child_obj_id, start_child_pose[:3], start_child_pose[3:]
            # )

            # # then, compute a new position + orientation for the parent object, that is upside down
            # upside_down_ori_mat = np.matmul(
            #     common.euler2rot([np.pi, 0, 0]), upright_parent_ori_mat
            # )
            # upside_down_pose_mat = np.eye(4)
            # upside_down_pose_mat[:-1, :-1] = upside_down_ori_mat
            # upside_down_pose_mat[:-1, -1] = start_parent_pose[:3]
            # upside_down_pose_mat[2, -1] += 0.15  # move up in z a bit
            # parent_upside_down_pose_list = util.pose_stamped2list(
            #     util.pose_from_matrix(upside_down_pose_mat)
            # )

            # # reset parent to this state and constrain to world
            # pb_client.reset_body(
            #     parent_obj_id,
            #     parent_upside_down_pose_list[:3],
            #     parent_upside_down_pose_list[3:],
            # )
            # ud_cid = constraint_obj_world(
            #     parent_obj_id,
            #     parent_upside_down_pose_list[:3],
            #     parent_upside_down_pose_list[3:],
            # )

            # # get the final relative pose of the child object
            # final_child_pose_parent = util.convert_reference_frame(
            #     pose_source=util.pose_from_matrix(final_child_pose_mat),
            #     pose_frame_target=util.pose_from_matrix(start_parent_pose_mat),
            #     pose_frame_source=util.unit_pose(),
            # )
            # # get the final world frame pose of the child object in upside down pose
            # final_child_pose_upside_down = util.convert_reference_frame(
            #     pose_source=final_child_pose_parent,
            #     pose_frame_target=util.unit_pose(),
            #     pose_frame_source=util.pose_from_matrix(upside_down_pose_mat),
            # )
            # final_child_pose_upside_down_list = util.pose_stamped2list(
            #     final_child_pose_upside_down
            # )
            # final_child_pose_upside_down_mat = util.matrix_from_pose(
            #     final_child_pose_upside_down
            # )

            # # reset child to this state
            # pb_client.reset_body(
            #     child_obj_id,
            #     final_child_pose_upside_down_list[:3],
            #     final_child_pose_upside_down_list[3:],
            # )

            # # turn on the simulation and wait for a couple seconds
            # pb_client.set_step_sim(False)
            # time.sleep(2.0)

            # # check if they are still in contact (they shouldn't be)
            # ud_obj_surf_contacts = p.getContactPoints(
            #     parent_obj_id, child_obj_id, -1, -1
            # )
            # ud_touching_surf = len(ud_obj_surf_contacts) > 0
            # success_crit_dict["fell_off_upside_down"] = not ud_touching_surf

            # #########################################################################

            # place_success = np.all(np.asarray(list(success_crit_dict.values())))

            obj_surf_contacts = p.getContactPoints(child_obj_id, parent_obj_id, -1, -1)
           

            touching_surf = len(obj_surf_contacts) > 0
            place_success = touching_surf and not intersecting

            place_success_list.append(place_success)
            log_str = "Iteration: %d, " % iteration

            kvs["Place Success"] = sum(place_success_list) / float(
                len(place_success_list)
            )

            if parent_class == "syn_container" and child_class == "bottle":
                kvs["Angle From Upright"] = angle_from_upright

            for k, v in kvs.items():
                log_str += "%s: %.3f, " % (k, v)
            for k, v in success_crit_dict.items():
                log_str += "%s: %s, " % (k, v)

            id_str = f", parent_id: {parent_id}, child_id: {child_id}"
            log_info(log_str + id_str)

            eval_iter_dir = osp.join(eval_save_dir, f"trial_{iteration}")
            util.safe_makedirs(eval_iter_dir)
            sample_fname = osp.join(
                eval_iter_dir, f"{exp_type}_success_rate_relation.npz"
            )
            full_cfg_fname = osp.join(eval_iter_dir, f"{exp_type}_full_config.json")
            results_txt_fname = osp.join(eval_iter_dir, f"{exp_type}_results.txt")
            full_cfg_dict["demo_program"] = demo_program
            np.savez(
                sample_fname,
                parent_id=parent_id,
                child_id=child_id,
                sweep=args.sweep,
                parent_train_ids=canon_target_training_ids,
                child_train_ids=canon_source_training_ids,
                is_parent_shapenet_obj=is_parent_shapenet_obj,
                is_child_shapenet_obj=is_child_shapenet_obj,
                success_criteria_dict=success_crit_dict,
                place_success=place_success,
                place_success_list=place_success_list,
                mesh_file=pc_master_dict["child"]["mesh_file"],
                demo_program=demo_program,
                args=args.__dict__,
                cfg=util.cn2dict(cfg),
            )
            json.dump(
                full_cfg_dict,
                open(full_cfg_fname, "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=4,
            )

            results_txt_dict = {}
            results_txt_dict["place_success"] = place_success
            results_txt_dict["place_success_list"] = place_success_list
            results_txt_dict["current_success_rate"] = sum(place_success_list) / float(
                len(place_success_list)
            )
            results_txt_dict["success_criteria_dict"] = success_crit_dict
            open(results_txt_fname, "w").write(str(results_txt_dict))

            eval_img_fname2 = osp.join(eval_iter_dir, f"{exp_type}_{iteration}.png")
            util.np2img(eval_rgb.astype(np.uint8), eval_img_fname2)

            pause_mc_thread(True)
            for pc in pcl:
                obj_id = pc_master_dict[pc]["pb_obj_id"]
                pb_client.remove_body(obj_id)
                recorder.remove_object(obj_id, mc_vis)
            mc_vis["scene/child_pcd_refine"].delete()
            mc_vis["scene/child_pcd_refine_1"].delete()
            mc_vis["scene/final_child_pcd"].delete()
            pause_mc_thread(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parent_class", type=str, required=True)
    parser.add_argument("--child_class", type=str, required=True)
    parser.add_argument("--rel_demo_exp", type=str, required=True)
    parser.add_argument("--rel_model_path", type=str, default=None)

    parser.add_argument("--canon_source_file_stamp", type=str, default=None)
    parser.add_argument("--canon_target_file_stamp", type=str, default=None)
    parser.add_argument("--demo_idx", type=int, default=0)

    parser.add_argument("--config", type=str, default="base_cfg")
    parser.add_argument("--exp", type=str, default="debug_eval")
    parser.add_argument("--eval_data_dir", type=str, default="eval_data")
    parser.add_argument(
        "--sweep",
        type=bool,
        default=False,
        help="If true tests on all valid object ids, not just those in test_ids. Shouldn't run with rndf",
    )

    parser.add_argument("--opt_visualize", action="store_true")
    parser.add_argument("--opt_iterations", type=int, default=100)
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--resume_iter", type=int, default=0)
    parser.add_argument(
        "--save_all_opt_results",
        action="store_true",
        help="If True, then we will save point clouds for all optimization runs, otherwise just save the best one (which we execute)",
    )
    parser.add_argument("--start_iteration", type=int, default=0)

    parser.add_argument("--single_instance", action="store_true")
    parser.add_argument("--rand_mesh_scale", action="store_true")
    parser.add_argument(
        "--parent_load_pose_type",
        type=str,
        default="demo_pose",
        help="Must be in [any_pose, demo_pose, random_upright]",
    )
    parser.add_argument(
        "--child_load_pose_type",
        type=str,
        default="demo_pose",
        help="Must be in [any_pose, demo_pose, random_upright]",
    )

    # rel ebm flags
    # parser.add_argument('--no_trans', action='store_true', help='whether or not to include translation opt')
    # parser.add_argument('--load_start', action='store_true', help='if we should load the start point clouds from demos')
    # parser.add_argument('--rand_pose', action='store_true')
    # parser.add_argument('--rand_rot', action='store_true')
    # parser.add_argument('--test_idx', default=0, type=int)
    # parser.add_argument('--real', action='store_true')

    parser.add_argument("--pybullet_viz", action="store_true")
    parser.add_argument("--pybullet_server", action="store_true")
    parser.add_argument("--is_parent_shapenet_obj", action="store_true")
    parser.add_argument("--is_child_shapenet_obj", action="store_true")
    parser.add_argument("--test_on_train", action="store_true")

    parser.add_argument(
        "--relation_method",
        type=str,
        default="intersection",
        help='either "intersection", "ebm"',
    )
    parser.add_argument(
        "--create_target_desc",
        action="store_true",
        help='If True and --relation_method="intersection", then create the target descriptors if a file does not already exist containing them',
    )
    parser.add_argument(
        "--target_desc_name", type=str, default="target_descriptors.npz"
    )
    parser.add_argument("--refine_with_ebm", action="store_true")
    parser.add_argument(
        "--pc_reference", type=str, default="parent", help='either "parent" or "child"'
    )
    parser.add_argument("--skip_alignment", action="store_true")
    parser.add_argument("--new_descriptors", action="store_true")
    parser.add_argument("--n_demos", type=int, default=0)
    parser.add_argument("--target_idx", type=int, default=-1)
    parser.add_argument("--query_scale", type=float, default=0.025)
    parser.add_argument("--target_rounds", type=int, default=3)

    # some threshold
    parser.add_argument("--upright_ori_diff_thresh", type=float, default=np.deg2rad(15))

    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--noise_idx", type=int, default=0)

    # custom
    parser.add_argument("--demo_selection", default=False, action="store_true")
    parser.add_argument("--ablate-no-warp", default=False, action="store_true")
    parser.add_argument("--ablate-no-scale", default=False, action="store_true")
    parser.add_argument("--ablate-no-pose-training", default=False, action="store_true")
    parser.add_argument("--ablate-no-size-reg", default=False, action="store_true")

    args = parser.parse_args()

    main(args)
