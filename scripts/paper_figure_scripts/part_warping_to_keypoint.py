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

#Load training mugs and models

inference_kwargs =  {
                            "train_latents": True,
                            "train_scales": True,
                            "train_poses": True,
                        }



parser = argparse.ArgumentParser()

parser.add_argument('-load_cached', action="store_true")    

args = parser.parse_args()


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

def get_segmented_rack_mesh(rack_mesh):
    pcl = rack_mesh.vertices

    #Get the top points 
    max_z = np.max(pcl[:,2]) - .01
    max_pts = pcl[pcl[:,2] > max_z]
    center = np.mean(max_pts, 0)

    #get the furthest out point on the branch
    max_r = np.argmax(np.linalg.norm(pcl[:, :2]-center[:2], axis=-1))
    max_r_multiple = np.argpartition(np.linalg.norm(pcl[:, :2], axis=-1), -2)[-2:]

    normal = pcl[max_r,:2]/np.linalg.norm(pcl[max_r,:2])
    normal_3d = np.array([normal[0], normal[1], 0])
    normal_angle = np.arctan2(normal[1], normal[0])

    rad = max(np.linalg.norm(max_pts[:, :2]-center[:2], axis=-1))

    v = rack_mesh.vertices
    f = rack_mesh.faces

    x = np.linspace(-.5, .5, 100)

    y = normal[0]/(-normal[1])*(x-(rad*np.cos(normal_angle)+center[0])) + (rad*np.sin(normal_angle)+center[1])

    import plotly.graph_objects as go

    fig = go.Figure(
    data=[  go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=f[:, 0],
            j=f[:, 1],
            k=f[:, 2],
            colorscale="Viridis",
            intensity = v[:, 2],), go.Scatter3d(x=x, y=y, z=[.5]*100),
            go.Scatter3d(x=pcl[max_r_multiple][:,0], y=pcl[max_r_multiple][:,1], z=pcl[max_r_multiple][:,2]), 
            go.Scatter3d(x=[pcl[max_r][0]], y=[pcl[max_r][1]], z=[pcl[max_r][2]]),
            go.Scatter3d(x=max_pts[:, 0], y=max_pts[:, 1], z=max_pts[:, 2], )]
        )
    #input("continue?")


    rack_trunk = trimesh.intersections.slice_mesh_plane(rack_mesh,
    -normal_3d,
    np.array([rad*np.cos(normal_angle)+center[0],rad*np.sin(normal_angle)+center[1],0]))
    #rack_trunk.show()

    rack_branch = trimesh.intersections.slice_mesh_plane(rack_mesh,
    normal_3d,
    np.array([rad*np.cos(normal_angle)+center[0],rad*np.sin(normal_angle)+center[1],0]))
    #rack_branch.show()

    
    # print("FACE COUNT")
    # print(len(rack_branch.faces))
    # print(len(rack_trunk.faces))
    # print()

    
    # rack_pts = utils.trimesh_create_verts_surface(rack_mesh, 1000)
    # trunk = rack_pts[np.linalg.norm(rack_pts[:, :2] - center[:2], axis=-1) < rad]
    # branch = rack_pts[np.linalg.norm(rack_pts[:, :2]- center[:2], axis=-1) > rad]

    return {1: rack_branch, 0:rack_trunk}


def get_z_descriptors(part_pcls, part_names):
    part_labels = {part: [] for part in part_names}
    for part in part_names: 
        z_mean = np.mean(part_pcls[part][:,2])
        
        part_labels[part].append(np.where(
                part_pcls[part][:,2] > z_mean,
                np.zeros_like(part_pcls[part][:, 0]),
                np.ones_like(part_pcls[part][:, 0]),
            ))
    return part_labels


def get_closest_points(target_point, pcl, n=10):
    distances = np.linalg.norm(target_point - pcl, axis=1)
    idxs = np.argpartition(distances, n)
    return idxs[:n]


if args.load_cached:
    cached = pickle.load(open('temp_data.pkl', 'rb'))
    globals().update(**cached)


obj_type = "mug"
part_labels = {"cup": 37, "handle": 36}
part_names = ["cup", "handle"]

directory = (
    "../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/"
)

training_ids = load_all_shapenet_files("mug")
test_id = training_ids[4]

mug_mesh = get_mesh(test_id)
part_meshes = get_segmented_mesh(test_id)

mug_part_pcls = {}

for part_id in part_meshes.keys():
    mug_part_pcls[part_names[part_id-37]] = utils.trimesh_create_verts_surface(part_meshes[part_id], 1000)

whole_mug_pcl = utils.trimesh_create_verts_surface(trimesh.util.concatenate(list(part_meshes.values())), 1000)

whole_mug_pcl,  mug_part_pcls['cup'],  mug_part_pcls['handle'] = utils.scale_points_circle([whole_mug_pcl,  mug_part_pcls['cup'],  mug_part_pcls['handle']], base_scale=.1)

handle_pts = get_closest_points(np.array([0,1,1]), mug_part_pcls['handle'])
cup_pts = get_closest_points(np.array([0,-1,1]), mug_part_pcls['cup'])


adjacent_part_pairs = [
    {"cup":  mug_part_pcls["cup"], "handle":  mug_part_pcls["handle"]},
]
ordered_part_pairs = [
    ("cup", "handle"),
]


part_mesh_vertices = {k: part_meshes[k].vertices for k in part_meshes.keys()}
part_mesh_faces = {k: part_meshes[k].faces for k in part_meshes.keys()}

whole_mesh = trimesh.util.concatenate(list(part_meshes.values()))
utils.trimesh_transform(whole_mesh, scale=.2, rotation=Rotation.from_euler('xyz', [0,0,0]).as_matrix())
whole_mesh_vertices = {'mug': whole_mesh.vertices}
whole_mesh_faces = {'mug': whole_mesh.faces}


mug_part_model_files = {'cup': '/home/rthomp12/fewshot/mug_cup_20250103-220631_5',
                        'handle': '/home/rthomp12/fewshot/mug_handle_20250103-220631_5', }

mug_whole_model_file = 'part_based_warp_models/whole_mug_20240501-191613_10'
mug_whole_canon_model =  utils.CanonPart.from_pickle(mug_whole_model_file)

mug_part_names = ['cup', 'handle']
mug_part_canon_models = {part: utils.CanonPart.from_pickle(mug_part_model_files[part]) for part in mug_part_names}

canon_adjacent_part_pairs = [
    {"cup": mug_part_canon_models["cup"], "handle": mug_part_canon_models["handle"]},
]

canon_handle_pt_idxs = get_closest_points(np.array([0,1,1]), mug_part_canon_models['handle'].canonical_pcl)
canon_cup_pt_idxs = get_closest_points(np.array([0,-1,1]), mug_part_canon_models['cup'].canonical_pcl)




whole_mug_pts = np.concatenate([get_closest_points(np.array([0,1,1]), mug_whole_canon_model.canonical_pcl), 
                                get_closest_points(np.array([0,-1,1]), mug_whole_canon_model.canonical_pcl), ])

canon_mug_labels = {}
mug_part_labels = {}
canon_mug_labels['relational'] = get_canon_labels(
                canon_adjacent_part_pairs, mug_part_canon_models, mug_part_names, 
            )
canon_mug_labels['variational'] =get_z_descriptors({'cup': mug_part_canon_models['cup'].canonical_pcl,
                                                    'handle': mug_part_canon_models['handle'].canonical_pcl,}, ['cup', 'handle'])


mug_part_labels['relational'] = get_part_labels(
    adjacent_part_pairs,
)
mug_part_labels['variational'] = get_z_descriptors(mug_part_pcls, ['cup', 'handle'])


device = 'cuda'
shape_weight = 1
variational_weight = .75


handle_pts_idxs = get_closest_points(np.array([0,1,1]), mug_part_pcls['handle'])
cup_pts_idxs = get_closest_points(np.array([0,-1,1]), mug_part_pcls['cup'])
canon_handle_pt_idxs = get_closest_points(np.array([0,1,1]), mug_part_canon_models['handle'].canonical_pcl)
canon_cup_pt_idxs = get_closest_points(np.array([0,-1,1]), mug_part_canon_models['cup'].canonical_pcl)
moveable_masks = {'cup': cup_pts_idxs, 'handle': handle_pts_idxs}
static_masks = {'cup': cup_pts_idxs, 'handle': handle_pts_idxs}


mug_mesh_reconstructions = {}
mug_reconstructions = {}
n_angles = 8
for target_part in mug_part_names:

    canon = mug_part_canon_models[target_part]
    canon_labels = {'variational': canon_mug_labels['variational'][target_part],
                    'relational': canon_mug_labels['relational'][target_part],}

    cost_function = (
        lambda source, target, canon_part_labels, 
            latent_param, scale_param, initial_latents: mask_and_cost_batch_pt(
            target,
            mug_part_labels['relational'][target_part],
            source,
            canon_part_labels['relational'], two_sided=(target_part=='handle' or target_part=='spout')
        ) + \
        mask_and_cost_batch_pt(
            target,
            mug_part_labels['variational'][target_part],
            source,
            canon_part_labels['variational'], two_sided=(target_part=='handle' or target_part=='spout')
        ) * variational_weight #+ torch.std(scale_param) * 50
    )

    warp = ObjectWarpingSE3Batch(
            canon,
            mug_part_pcls[target_part],
            'cuda',
            canon_labels=canon_labels,
            cost_function=cost_function,
            **cp.deepcopy(PARAM_1),
        )
    mug_reconstructions[target_part], _, part_param = warp_to_pcd_se3(
        warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
    )
    mug_mesh_reconstructions[target_part] = mug_part_canon_models[target_part].to_transformed_mesh(part_param)

    utils.generate_slider_viz(
        warp, {'Target': mug_part_pcls[target_part]}, 
        {f'{target_part}': mug_whole_canon_model.canonical_pcl},  
        use_latents=True, model=canon, 
        static_masks={target_part: static_masks[target_part]}, tf_mask={moveable_part: moveable_masks[target_part]}, generate_animation=True, 
        experiment_id=f'scripts/paper_figure_scripts/keypoint_{target_part}_warping'
    ).show()


exit(0)


obj_type = "syn_rack_easy"
part_names = ["trunk", "branch"]

directory = (
    "../relational_ndf/src/rndf_robot/descriptions/objects/syn_racks_easy_obj/"
)


training_ids = load_all_shapenet_files("syn_rack_easy")
test_id = training_ids[4]

tree_mesh = get_rack_mesh(test_id)
part_meshes = get_segmented_rack_mesh(get_rack_mesh(test_id))

tree_part_pcls = {}

for part_id in part_meshes.keys():
    tree_part_pcls[part_names[part_id]] = utils.trimesh_create_verts_surface(part_meshes[part_id], 1000)

whole_tree_pcl = utils.trimesh_create_verts_surface(trimesh.util.concatenate(list(part_meshes.values())), 1000)
whole_tree_pcl, tree_part_pcls['trunk'], tree_part_pcls['branch'] = utils.scale_points_circle([whole_tree_pcl, tree_part_pcls['trunk'], tree_part_pcls['branch']], base_scale=.1)

branch_pts = tree_part_pcls['branch'][get_closest_points(np.array([0,2,2]), tree_part_pcls['branch'])]
trunk_pts = tree_part_pcls['trunk'][get_closest_points(np.array([0,0,0]), tree_part_pcls['trunk'])]


adjacent_part_pairs = [
    {"trunk": tree_part_pcls["trunk"], "branch": tree_part_pcls["branch"]},
]
ordered_part_pairs = [
    ("trunk", "branch"),
]

tree_part_model_files = {'trunk': 'part_based_warp_models/trunk_20240430-043520_10', 
                         'branch': 'part_based_warp_models/branch_20240430-043520_10', }

tree_whole_model_file = 'part_based_warp_models/whole_syn_rack_easy_20240430-043520_10'
tree_whole_canon_model =  utils.CanonPart.from_pickle(tree_whole_model_file)

tree_part_names = ['trunk', 'branch']
tree_part_canon_models = {part: utils.CanonPart.from_pickle(tree_part_model_files[part]) for part in tree_part_names}


canon_adjacent_part_pairs = [
    {"trunk": tree_part_canon_models["trunk"], "branch": tree_part_canon_models["branch"]},
]


canon_branch_pt_idxs = get_closest_points(np.array([0,2,2]), tree_part_canon_models['branch'].canonical_pcl)
canon_trunk_pt_idxs = get_closest_points(np.array([0,0,0]), tree_part_canon_models['trunk'].canonical_pcl)

whole_tree_pts = np.concatenate([get_closest_points(np.array([0,-1,.1]), tree_whole_canon_model.canonical_pcl), 
                                get_closest_points(np.array([0,0,0]), tree_whole_canon_model.canonical_pcl), ])

canon_tree_labels = {}
tree_part_labels = {}
canon_tree_labels['relational'] = get_canon_labels(
                canon_adjacent_part_pairs, tree_part_canon_models, tree_part_names, 
            )
canon_tree_labels['variational'] =get_z_descriptors({'trunk': tree_part_canon_models['trunk'].canonical_pcl,
                                                    'branch': tree_part_canon_models['branch'].canonical_pcl,}, ['trunk', 'branch'])


tree_part_labels['relational'] = get_part_labels(
    adjacent_part_pairs,
)
tree_part_labels['variational'] = get_z_descriptors(tree_part_pcls, ['trunk', 'branch'])



tree_mesh_reconstructions = {}
tree_reconstructions = {}
n_angles = 8
for target_part in tree_part_names:

    canon = tree_part_canon_models[target_part]
    canon_labels = {'variational': canon_tree_labels['variational'][target_part],
                    'relational': canon_tree_labels['relational'][target_part],}

    cost_function = (
        lambda source, target, canon_part_labels, 
            latent_param, scale_param, initial_latents: mask_and_cost_batch_pt(
            target,
            tree_part_labels['relational'][target_part],
            source,
            canon_part_labels['relational'], two_sided=(target_part=='branch' or target_part=='spout')
        ) + \
        mask_and_cost_batch_pt(
            target,
            tree_part_labels['variational'][target_part],
            source,
            canon_part_labels['variational'], two_sided=(target_part=='branch' or target_part=='spout')
        ) * .75 #+ torch.std(scale_param) * 50
    )

    warp = ObjectWarpingSE3Batch(
            canon,
            tree_part_pcls[target_part],
            'cuda',
            canon_labels=canon_labels,
            cost_function=cost_function,
            **cp.deepcopy(PARAM_1),
        )
    tree_reconstructions[target_part], _, part_param = warp_to_pcd_se3(
        warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
    )
    tree_mesh_reconstructions[target_part] = tree_part_canon_models[target_part].to_transformed_mesh(part_param)

    utils.generate_slider_viz(
        warp, {'Target': tree_part_pcls[target_part]}, 
        {f'{target_part}': tree_whole_canon_model.canonical_pcl},  
        use_latents=True, model=canon, 
        static_masks=None, tf_mask=None, generate_animation=True, 
        experiment_id=f'scripts/paper_figure_scripts/{target_part}_warping'
    ).show()