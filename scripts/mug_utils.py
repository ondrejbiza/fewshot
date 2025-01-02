from src import viz_utils, utils
import trimesh
import os, os.path as osp
from rndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from rndf_robot.utils import util, path_util
from airobot import log_info
from airobot.utils import common
from rndf_robot.share.globals import (
    bad_shapenet_mug_ids_list,
    bad_shapenet_bowls_ids_list,
    bad_shapenet_bottles_ids_list,
)
import numpy as np
import copy as cp
from scipy.spatial.transform import Rotation
from src.ndf_interface import get_part_labels, mask_and_cost_batch_pt, get_canon_labels, get_part_labels
from src.object_warping import ObjectWarpingSE3Batch, PARAM_1, warp_to_pcd_se3
from sklearn import neighbors
import argparse
import pickle

def load_segmented_pointcloud_from_txt(pcl_id, num_points=2048, 
                                       root = 'Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390'):
    fn = os.path.join(root, pcl_id + '.txt')
    cls = 'Mug'
    data = np.loadtxt(fn).astype(np.float32)
    point_set = data[:, 0:3]#<-- ignore the normals
    seg_ids = data[:, -1].astype(np.int32)
    point_set[:, 0:3] = utils.center_pcl(point_set[:, 0:3])

    #fixed transform to align with the other mugs being used
    rotation = Rotation.from_euler("zyx", [0., np.pi/2, 0.]).as_quat()
    transform = utils.pos_quat_to_transform([0,0,0], rotation)
    point_set = utils.transform_pcd(point_set, transform)
    #point_set = utils.scale_points_circle([point_set], base_scale=0.1)[0]
   
    return point_set, cls, seg_ids


def load_all_shapenet_files(obj_type):
    cfg = get_eval_cfg_defaults()
    config_fname = osp.join(
        path_util.get_rndf_config(), "eval_cfgs", "base_cfg"
    )  # args.config)
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        log_info(f"Config file {config_fname} does not exist, using defaults")

    mesh_data_dirs = {
        "mug": "mug_centered_obj_normalized",
        # 'bottle': 'bottle_centered_obj_normalized',
        "bowl": "bowl_centered_obj_normalized",
        "syn_rack_easy": "syn_racks_easy_obj",
        # 'syn_container': 'box_containers_unnormalized'
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
        "syn_container": [],
    }

    upright_orientation_dict = {
        "mug": common.euler2quat([np.pi / 2, 0, 0]).tolist(),
        "bottle": common.euler2quat([np.pi / 2, 0, 0]).tolist(),
        "bowl": common.euler2quat([np.pi / 2, 0, 0]).tolist(),
        "syn_rack_easy": common.euler2quat([0, 0, 0]).tolist(),
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

        # train_objects = sorted(objects_filtered)[:train_n]
        # test_objects = sorted(objects_filtered)[train_n:]

        # log_info('\n\n\nTest objects: ')
        # log_info(test_objects)
        # # log_info('\n\n\n')

        mesh_names[k] = objects_filtered

    obj_classes = list(mesh_names.keys())

    scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
    scale_default = cfg.MESH_SCALE_DEFAULT

    # cfg.OBJ_SAMPLE_Y_HIGH_LOW = [0.3, -0.3]
    cfg.OBJ_SAMPLE_Y_HIGH_LOW = [-0.35, 0.175]
    x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
    y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
    table_z = cfg.TABLE_Z

    return mesh_names[obj_type]


def get_mesh(pcl_id):
    obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
    mesh = utils.trimesh_load_object(obj_file_path)
    return mesh

def get_rack_mesh(pcl_id, obj_file_path="../relational_ndf/src/rndf_robot/descriptions/objects/syn_racks_easy_obj/"):
    mesh = utils.trimesh_load_object(obj_file_path + f"{pcl_id}")
    return mesh

def get_segmented_mesh(pcl_id):
    seg_pcl, _, seg_ids = load_segmented_pointcloud_from_txt(pcl_id)
    obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
    unseg_mesh = utils.trimesh_load_object(obj_file_path)
    rotation = Rotation.from_euler('xyz', [np.pi/2,0,np.pi/8]).as_matrix()

    utils.trimesh_transform(unseg_mesh, center=True, scale=None, rotation=rotation)#Rotation.from_euler('xyz', [0, np.pi/2, 0]).to_mat())

    seg_rot = utils.pos_quat_to_transform([0,0,0],  Rotation.from_euler('xyz', [np.pi/2,0,np.pi/8]).as_quat())
    seg_pcl = utils.transform_pcd(seg_pcl, seg_rot)
    unseg_pcl, _ = utils.trimesh_get_vertices_and_faces(unseg_mesh)

    X = seg_pcl
    Y = seg_ids
    clf = neighbors.KNeighborsClassifier(2)#svm.SVC()
    clf.fit(X, Y)

    part_meshes = {}
    for part_label in clf.classes_:
        part_mesh = cp.deepcopy(unseg_mesh)
        unseg_ids = clf.predict(unseg_pcl)
        mask = unseg_ids==part_label
        face_mask = mask[part_mesh.faces].all(axis=1)
        part_mesh.update_faces(face_mask)
        part_mesh.remove_unreferenced_vertices()
        part_meshes[part_label] = part_mesh
    return part_meshes
