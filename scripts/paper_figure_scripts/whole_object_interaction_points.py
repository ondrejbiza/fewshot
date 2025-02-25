



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
from src import demo
from src.ndf_interface import get_part_labels, mask_and_cost_batch_pt, get_canon_labels, get_part_labels
from src.object_warping import ObjectWarpingSE3Batch, PARAM_1, warp_to_pcd_se3
from sklearn import neighbors
import argparse
import pickle

#Load training mugs and models

inference_kwargs =  {
                            "train_latents": True,
                            "train_scales": True,
                            "train_poses": True,
                        }



parser = argparse.ArgumentParser()

parser.add_argument('-load_cached', action="store_true")    

args = parser.parse_args()


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


def get_closest_points(target_point, pcl, n=10):
    distances = np.linalg.norm(target_point - pcl, axis=1)
    idxs = np.argpartition(distances, n)
    return idxs[:n]





obj_type = "mug"
directory = (
    "../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/"
)

training_ids = load_all_shapenet_files("mug")
test_id = training_ids[0]

mug_mesh = get_mesh(test_id)


whole_mesh = mug_mesh
utils.trimesh_transform(whole_mesh, scale=.2, rotation=Rotation.from_euler('xyz', [np.pi/2,0,0]).as_matrix())
whole_mug_pcl = utils.trimesh_create_verts_surface(whole_mesh, 1000)

whole_mug_pcl = utils.scale_points_circle([whole_mug_pcl], base_scale=.1)[0]

whole_mesh_vertices = {'mug': whole_mesh.vertices}
whole_mesh_faces = {'mug': whole_mesh.faces}



mug_whole_model_file = 'part_based_warp_models/whole_mug_20240501-191613_10'
mug_whole_canon_model =  utils.CanonPart.from_pickle(mug_whole_model_file)


whole_mug_pts = np.concatenate([get_closest_points(np.array([0,1,1]), mug_whole_canon_model.canonical_pcl), 
                                get_closest_points(np.array([0,-1,1]), mug_whole_canon_model.canonical_pcl), ])

whole_mug_reconstruction = None
n_angles = 8
warp = ObjectWarpingSE3Batch(
            mug_whole_canon_model,
            whole_mug_pcl,
            'cuda',
            **cp.deepcopy(PARAM_1),
        )

whole_mug_reconstruction, _, whole_mug_params = warp_to_pcd_se3(
        warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
    )


obj_type = "syn_rack_easy"
directory = (
    "../relational_ndf/src/rndf_robot/descriptions/objects/syn_racks_easy_obj/"
)


training_ids = load_all_shapenet_files("syn_rack_easy")
test_id = training_ids[4]

tree_mesh = get_rack_mesh(test_id)

whole_tree_pcl = utils.trimesh_create_verts_surface(tree_mesh, 1000)
whole_tree_pcl= utils.scale_points_circle([whole_tree_pcl], base_scale=.1)[0]

tree_whole_model_file = 'part_based_warp_models/whole_syn_rack_easy_20240430-043520_10'
tree_whole_canon_model =  utils.CanonPart.from_pickle(tree_whole_model_file)



whole_tree_pts = np.concatenate([get_closest_points(np.array([0,-1,.1]), tree_whole_canon_model.canonical_pcl), 
                                get_closest_points(np.array([0,0,0]), tree_whole_canon_model.canonical_pcl), ])


whole_tree_reconstruction = None

n_angles = 15

warp = ObjectWarpingSE3Batch(
            tree_whole_canon_model,
            whole_tree_pcl,
            'cuda',
            **cp.deepcopy(PARAM_1),
        )

whole_tree_reconstruction, _, whole_tree_params = warp_to_pcd_se3(
        warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
    )

def get_interaction_points(child_params, parent_params, child_model, parent_model):
    nearby_points_delta = 0.03 # Empirically picked

    (
        knns,
        deltas,
        target_indices,
    ) = demo.save_place_nearby_points_v2(
        whole_mug_pcl,
        whole_tree_pcl,
        child_model,
        child_params,
        parent_model,
        parent_params,
        nearby_points_delta,
    )

    anchors = child_model.canonical_pcl[
        knns]
    targets_child = np.mean(
        anchors + deltas, axis=1
    )
    targets_parent = parent_model.canonical_pcl[
        target_indices
    ] 

    viz_utils.show_pcds_plotly({'child': child_model.canonical_pcl, 
                                'parent': parent_model.canonical_pcl, 
                                'child_targets': targets_child, 
                                'parent_targets':targets_parent},axis_visible=False).show()

get_interaction_points(whole_mug_params, whole_tree_params, mug_whole_canon_model, tree_whole_canon_model, )