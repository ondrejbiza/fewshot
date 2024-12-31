from src import viz_utils, utils
import trimesh
import os, os.path as osp
from rndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from rndf_robot.utils import util, path_util
from airobot import log_info
from airobot.utils import common
from rndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
import numpy as np


def load_all_shapenet_files(obj_type):
    cfg = get_eval_cfg_defaults()
    config_fname = osp.join(path_util.get_rndf_config(), 'eval_cfgs', 'base_cfg')#args.config)
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        log_info(f'Config file {config_fname} does not exist, using defaults')

    mesh_data_dirs = {
        'mug': 'mug_centered_obj_normalized',
        # 'bottle': 'bottle_centered_obj_normalized',
        'bowl': 'bowl_centered_obj_normalized',
        'syn_rack_easy': 'syn_racks_easy_obj',
        # 'syn_container': 'box_containers_unnormalized'
    }
    mesh_data_dirs = {k: osp.join(path_util.get_rndf_obj_descriptions(), v) for k, v in mesh_data_dirs.items()}
    bad_ids = {
        'syn_rack_easy': [],
        'bowl': bad_shapenet_bowls_ids_list,
        'mug': bad_shapenet_mug_ids_list,
        'bottle': bad_shapenet_bottles_ids_list,
        'syn_container': []
    }

    upright_orientation_dict = {
        'mug': common.euler2quat([np.pi/2, 0, 0]).tolist(),
        'bottle': common.euler2quat([np.pi/2, 0, 0]).tolist(),
        'bowl': common.euler2quat([np.pi/2, 0, 0]).tolist(),
        'syn_rack_easy': common.euler2quat([0, 0, 0]).tolist(),
        'syn_container': common.euler2quat([0, 0, 0]).tolist(),
    }

    print("Iterating mesh data")
    mesh_names = {}
    for k, v in mesh_data_dirs.items():
        # get train samples
        objects_raw = os.listdir(v) 
        objects_filtered = [fn for fn in objects_raw if (fn.split('/')[-1] not in bad_ids[k] and '_dec' not in fn)]
        # objects_filtered = objects_raw
        total_filtered = len(objects_filtered)
        train_n = int(total_filtered * 0.9); test_n = total_filtered - train_n

        # train_objects = sorted(objects_filtered)[:train_n]
        # test_objects = sorted(objects_filtered)[train_n:]

        # log_info('\n\n\nTest objects: ')
        # log_info(test_objects)
        # # log_info('\n\n\n')

        mesh_names[k] = objects_filtered
    print("got em")

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



obj_type = 'mug'
part_labels = {'cup': 37, 'handle': 36}
part_names = ['cup', 'handle']
directory = "../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/"


training_ids = load_all_shapenet_files('mug')
mug_meshes = {'mug': []}
for obj_id in training_ids[:1]:
    mug_meshes['mug'].append(get_mesh(obj_id))


root = '~/fewshot/segmentations/'
for i in range(5):
	parts = ['body', 'lid', 'spout', 'handle']

	mesh_file = {part: f'kettle_{i+1}/kettle_{i+1}_{part}.obj' for part in parts}
	meshes = {part: trimesh.load(root + mesh_file[part]) for part in parts}

	for part in parts:
		utils.trimesh_transform(meshes[part], center=False, scale=1.75)

	viz_utils.show_meshes_plotly({part:meshes[part].vertices for part in parts}|{'mug': mug_meshes['mug'][0].vertices}, {part:meshes[part].faces for part in parts}|{'mug': mug_meshes['mug'][0].faces})