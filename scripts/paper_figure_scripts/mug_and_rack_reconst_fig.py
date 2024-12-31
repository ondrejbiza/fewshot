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

handle_pts = mug_part_pcls['handle'][get_closest_points(np.array([0,1,1]), mug_part_pcls['handle'])]
cup_pts = mug_part_pcls['cup'][get_closest_points(np.array([0,-1,1]), mug_part_pcls['cup'])]

adjacent_part_pairs = [
    {"cup":  mug_part_pcls["cup"], "handle":  mug_part_pcls["handle"]},
]
ordered_part_pairs = [
    ("cup", "handle"),
]


# colors = {
#     "cup_handle_0_cup": "gray",
#     "cup_handle_1_cup": "emrld",
#     "cup_handle_0_handle": "gray",
#     "cup_handle_1_handle": "purp",
# }

part_mesh_vertices = {k: part_meshes[k].vertices for k in part_meshes.keys()}
part_mesh_faces = {k: part_meshes[k].faces for k in part_meshes.keys()}

whole_mesh = trimesh.util.concatenate(list(part_meshes.values()))
utils.trimesh_transform(whole_mesh, scale=.2, rotation=Rotation.from_euler('xyz', [0,0,0]).as_matrix())
whole_mesh_vertices = {'mug': whole_mesh.vertices}
whole_mesh_faces = {'mug': whole_mesh.faces}


mug_part_model_files = {'cup': 'part_based_warp_models/cup_20240501-191613_10', 
                        'handle': 'part_based_warp_models/handle_20240501-191613_10', }

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


canon_mug_labels = get_canon_labels(
                canon_adjacent_part_pairs, mug_part_canon_models, mug_part_names, 
            )

split_canon_mug_pcls = { "canon_cup_0": mug_part_canon_models["cup"].canonical_pcl[canon_mug_labels['cup'][0] == 0], 
                         "canon_cup_1": mug_part_canon_models["cup"].canonical_pcl[canon_mug_labels['cup'][0] == 1],
                         "canon_handle_0": mug_part_canon_models["handle"].canonical_pcl[canon_mug_labels['handle'][0] == 0],
                         "canon_handle_1": mug_part_canon_models["handle"].canonical_pcl[canon_mug_labels['handle'][0] == 1]
}

mug_part_labels = get_part_labels(
    adjacent_part_pairs,
)


if not args.load_cached:
    whole_mug_reconstruction = None
    mug_reconstructions = {part:None for part in mug_part_names}


    n_angles = 15


    warp = ObjectWarpingSE3Batch(
                mug_whole_canon_model,
                whole_mug_pcl,
                'cuda',
                **cp.deepcopy(PARAM_1),
            )

    whole_mug_reconstruction, _, whole_mug_params = warp_to_pcd_se3(
            warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
        )

    whole_mesh_reconstruction = mug_whole_canon_model.to_transformed_mesh(whole_mug_params)

    #utils.trimesh_transform(whole_mesh_reconstruction, scale=5, )#rotation=Rotation.from_euler('xyz', [0,0,np.pi]).as_matrix())
    # viz_utils.show_pcds_plotly({'target_mug': whole_mug_pcl, 'canon_mug':  mug_whole_canon_model.canonical_pcl, 'mug': whole_mug_reconstruction}).show()
    WHOLE_MUG_MESH = viz_utils.show_meshes_plotly({'whole': whole_mesh_reconstruction.vertices} | whole_mesh_vertices, 
                                                  {'whole': whole_mesh_reconstruction.faces} | whole_mesh_faces, 
                                                  pcds = {'cup': cup_pts, 'handle': handle_pts}, 
                                                  axis_visible=False)
    WHOLE_MUG_MESH.show()

    mug_mesh_reconstructions = {}
    n_angles = 15
    for target_part in mug_part_names:

        canon = mug_part_canon_models[target_part]

        cost_function = (
            lambda source, target, canon_part_labels: mask_and_cost_batch_pt(
                target,
                mug_part_labels[target_part],
                source,
                canon_part_labels,
            )
        )
        
        warp = ObjectWarpingSE3Batch(
                canon,
                mug_part_pcls[target_part],
                'cuda',
                canon_labels=canon_mug_labels[target_part],
                cost_function=cost_function,
                **cp.deepcopy(PARAM_1),
            )
        mug_reconstructions[target_part], _, part_param = warp_to_pcd_se3(
            warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
        )
        mug_mesh_reconstructions[target_part] = mug_part_canon_models[target_part].to_transformed_mesh(part_param)



    # viz_utils.show_pcds_plotly({'target_mug': whole_mug_pcl, 'canon_mug':  mug_whole_canon_model.canonical_pcl, 'mug': whole_mug_reconstruction}).show()



    # viz_utils.show_pcds_plotly({'target_mug': whole_pcl, 'canon_mug':  mug_whole_canon_model.canonical_pcl,} | {part: mug_reconstructions[part] for part in mug_part_names}).show()


    no_rel_mug_reconstructions = {part:None for part in mug_part_names}
    no_rel_mug_mesh_reconstructions = {}

    n_angles = 15
    for target_part in mug_part_names:

        canon = mug_part_canon_models[target_part]
        warp = ObjectWarpingSE3Batch(
                canon,
                mug_part_pcls[target_part],
                'cuda',
                **cp.deepcopy(PARAM_1),
            )
        no_rel_mug_reconstructions[target_part], _, part_param = warp_to_pcd_se3(
            warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
        )
        no_rel_mug_mesh_reconstructions[target_part] = mug_part_canon_models[target_part].to_transformed_mesh(part_param)



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

# all_labeled_parts = {}
# for part_pair, ordered_part_pair in zip(adjacent_part_pairs, ordered_part_pairs):
#     part_labels = get_part_labels([part_pair])

#     all_labeled_parts = (
#         all_labeled_parts
#         | {
#             f"{ordered_part_pair[0]}_{ordered_part_pair[1]}_0_{part}": part_pcls[part][
#                 part_labels[part][0] == 0
#             ]
#             for part in part_pair.keys()
#         }
#         | {
#             f"{ordered_part_pair[0]}_{ordered_part_pair[1]}_1_{part}": part_pcls[part][
#                 part_labels[part][0] == 1
#             ]
#             for part in part_pair.keys()
#         }
#     )

# colors = {
#     "trunk_branch_0_trunk": "gray",
#     "trunk_branch_1_trunk": "emrld",
#     "trunk_branch_0_branch": "gray",
#     "trunk_branch_1_branch": "purp",
# }

# viz_utils.show_pcds_plotly(all_labeled_parts, colors=colors).show()

tree_part_model_files = {'trunk': 'part_based_warp_models/trunk_20240430-043520_10', 
                         'branch': 'part_based_warp_models/branch_20240430-043520_10', }

tree_whole_model_file = 'part_based_warp_models/whole_syn_rack_easy_20240430-043520_10'
tree_whole_canon_model =  utils.CanonPart.from_pickle(tree_whole_model_file)

tree_part_names = ['trunk', 'branch']
tree_part_canon_models = {part: utils.CanonPart.from_pickle(tree_part_model_files[part]) for part in tree_part_names}


canon_adjacent_part_pairs = [
    {"trunk": tree_part_canon_models["trunk"], "branch": tree_part_canon_models["branch"]},
]
canon_tree_labels = get_canon_labels(
                canon_adjacent_part_pairs, tree_part_canon_models, tree_part_names, 
            )


canon_branch_pt_idxs = get_closest_points(np.array([0,2,2]), tree_part_canon_models['branch'].canonical_pcl)
canon_trunk_pt_idxs = get_closest_points(np.array([0,0,0]), tree_part_canon_models['trunk'].canonical_pcl)

whole_tree_pts = np.concatenate([get_closest_points(np.array([0,-1,.1]), tree_whole_canon_model.canonical_pcl), 
                                get_closest_points(np.array([0,0,0]), tree_whole_canon_model.canonical_pcl), ])

split_canon_tree_pcls = { "canon_trunk_0": tree_part_canon_models["trunk"].canonical_pcl[canon_tree_labels['trunk'][0] == 0], 
                         "canon_trunk_1": tree_part_canon_models["trunk"].canonical_pcl[canon_tree_labels['trunk'][0] == 1],
                         "canon_branch_0": tree_part_canon_models["branch"].canonical_pcl[canon_tree_labels['branch'][0] == 0],
                         "canon_branch_1": tree_part_canon_models["branch"].canonical_pcl[canon_tree_labels['branch'][0] == 1]
}



tree_part_labels = get_part_labels(
    adjacent_part_pairs,
)


if not args.load_cached:
    tree_reconstructions = {part:None for part in tree_part_names}

    whole_tree_reconstruction = None

    n_angles = 15


    warp = ObjectWarpingSE3Batch(
                tree_whole_canon_model,
                whole_tree_pcl,
                'cuda',
                **cp.deepcopy(PARAM_1),
            )

    whole_tree_reconstruction, _, _ = warp_to_pcd_se3(
            warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
        )


    viz_utils.show_pcds_plotly({'target_tree': whole_tree_pcl, 'tree': whole_tree_reconstruction}).show()

    n_angles = 15
    for target_part in tree_part_names:
        canon = tree_part_canon_models[target_part]

        cost_function = (
            lambda source, target, canon_part_labels: mask_and_cost_batch_pt(
                target,
                tree_part_labels[target_part],
                source,
                canon_part_labels,
            )
        )
        
        warp = ObjectWarpingSE3Batch(
                canon,
                tree_part_pcls[target_part],
                'cuda',
                canon_labels=canon_tree_labels[target_part],
                cost_function=cost_function,
                **cp.deepcopy(PARAM_1),
            )
        tree_reconstructions[target_part], _, _ = warp_to_pcd_se3(
            warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
        )

    # viz_utils.show_pcds_plotly({'target_tree': whole_pcl,} | {part: tree_reconstructions[part] for part in tree_part_names}).show()


    no_rel_tree_reconstructions = {part:None for part in tree_part_names}

    n_angles = 15
    for target_part in tree_part_names:

        canon = tree_part_canon_models[target_part]
        warp = ObjectWarpingSE3Batch(
                canon,
                tree_part_pcls[target_part],
                'cuda',
                **cp.deepcopy(PARAM_1),
            )
        no_rel_tree_reconstructions[target_part], _, _ = warp_to_pcd_se3(
            warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
        )


# viz_utils.show_pcds_plotly({'target_tree': whole_pcl, } | {part: no_rel_tree_reconstructions[part] for part in tree_part_names}).show()



target_marker={
            "size": 5,
            "opacity": .5
        }

source_marker = {
            "size": 5,
            "opacity": .9
        }

pickle.dump({
    'whole_mug_reconstruction': whole_mug_reconstruction, 
    'whole_tree_reconstruction': whole_tree_reconstruction,
    'no_rel_tree_reconstructions': no_rel_tree_reconstructions,
    'no_rel_mug_reconstructions': no_rel_mug_reconstructions,
    "mug_reconstructions": mug_reconstructions,
    "tree_reconstructions": tree_reconstructions,
    }, open('temp_data.pkl', 'wb'), )


# TARGET_MUG_MESH = viz_utils.show_meshes_plotly(whole_mesh_vertices, 
#                                                whole_mesh_faces, 
#                                                pcds = {'cup': cup_pts, 'handle': handle_pts}, 
#                                                axis_visible=False)
# PART_MUG_MESH = viz_utils.show_meshes_plotly({'cup': mug_mesh_reconstructions['cup'].vertices, 
#                                               'handle': mug_mesh_reconstructions['handle'].vertices} | whole_mesh_vertices, 
#                                               {'cup': mug_mesh_reconstructions['cup'].faces, 
#                                                'handle': mug_mesh_reconstructions['handle'].faces} | whole_mesh_faces, 
#                                               pcds = {'cup': cup_pts, 'handle': handle_pts}, 
#                                               axis_visible=False)
# NO_REL_PART_MUG_MESH = viz_utils.show_meshes_plotly({'cup': no_rel_mug_mesh_reconstructions['cup'].vertices, 'handle': no_rel_mug_mesh_reconstructions['handle'].vertices} | whole_mesh_vertices, 
#                                                   {'cup': no_rel_mug_mesh_reconstructions['cup'].faces, 'handle': no_rel_mug_mesh_reconstructions['handle'].faces} | whole_mesh_faces, 
#                                                   pcds = {'cup': cup_pts, 'handle': handle_pts}, 
#                                                   axis_visible=False)


named_figs = {
    "target_mug": {"whole_target": whole_mug_pcl, 'target_handle_pts': handle_pts, 'target_cup_pts': cup_pts},
    "target_rack": {"whole_target": whole_tree_pcl, 'target_branch_pts': branch_pts, 'target_trunk_pts': trunk_pts},
    "whole_mug": { 
                   "whole_target": whole_mug_pcl,
                   "whole_source": whole_mug_reconstruction,
                   "whole_mug_pts": whole_mug_reconstruction[whole_mug_pts],
                  },
    "whole_rack": {
                   "whole_target": whole_tree_pcl,
                   "whole_source": whole_tree_reconstruction,
                   "whole_tree_pts": whole_tree_reconstruction[whole_tree_pts],
                  },
    "no_rel_mug": {
                   "whole_target": whole_mug_pcl,
                   "source_cup": no_rel_mug_reconstructions['cup'],
                   "source_handle": no_rel_mug_reconstructions['handle'],
                   'source_cup_pts': no_rel_mug_reconstructions['cup'][canon_cup_pt_idxs], 
                   'source_handle_pts': no_rel_mug_reconstructions['handle'][canon_handle_pt_idxs], 
                  },
    "no_rel_rack": {
                    "whole_target": whole_tree_pcl,
                    "source_trunk":  no_rel_tree_reconstructions['trunk'],
                    "source_handle": no_rel_tree_reconstructions['branch'],
                    "source_trunk_pts":  no_rel_tree_reconstructions['trunk'][canon_trunk_pt_idxs],
                    "source_branch_pts": no_rel_tree_reconstructions['branch'][canon_branch_pt_idxs],
                    },
    "rel_mug": {
                "whole_target": whole_mug_pcl,
                "source_cup_0": mug_reconstructions['cup'][canon_mug_labels['cup'][0]==0],
                "source_cup_1": mug_reconstructions['cup'][canon_mug_labels['cup'][0]==1],
                "source_handle_0": mug_reconstructions['handle'][canon_mug_labels['handle'][0]==0],
                "source_handle_1": mug_reconstructions['handle'][canon_mug_labels['handle'][0]==1],
                'source_cup_pts': mug_reconstructions['cup'][canon_cup_pt_idxs], 
                'source_handle_pts': mug_reconstructions['handle'][canon_handle_pt_idxs], 
                },
    "rel_rack": {
                "whole_target": whole_tree_pcl,
                "source_trunk_0": tree_reconstructions['trunk'][canon_tree_labels['trunk'][0]==0],
                "source_trunk_1": tree_reconstructions['trunk'][canon_tree_labels['trunk'][0]==1],
                "source_branch_0": tree_reconstructions['branch'][canon_tree_labels['branch'][0]==0],
                "source_branch_1": tree_reconstructions['branch'][canon_tree_labels['branch'][0]==1],
                "source_trunk_pts":  tree_reconstructions['trunk'][canon_trunk_pt_idxs],
                "source_branch_pts": tree_reconstructions['branch'][canon_branch_pt_idxs],
                },
}

named_markers = {
    "target_mug": { "whole_target": target_marker | {"colorscale": "teal"},
                   'target_handle_pts': target_marker | {"colorscale": "reds"}, 
                   'target_cup_pts': target_marker | {"colorscale": "reds"}},
                 #  "canon_cup_1":    source_marker | {"colorscale": "emrld"}, 
                 #  "canon_cup_0":    source_marker | {"colorscale": "gray"},
                 #  "canon_handle_1": source_marker | {"colorscale": "purp"},
                 #  "canon_handle_0": source_marker | {"colorscale": "gray"}
                 # },
    "target_rack": { "whole_target": target_marker | {"colorscale": "teal"},
                    'target_branch_pts': target_marker | {"colorscale": "reds"}, 
                    'target_trunk_pts': target_marker | {"colorscale": "reds"} },
                  #  "canon_trunk_1":  source_marker | {"colorscale": "emrld"}, 
                  #  "canon_trunk_0":  source_marker | {"colorscale": "gray"},
                  #  "canon_branch_1": source_marker | {"colorscale": "purp"},
                  #  "canon_branch_0": source_marker | {"colorscale": "gray"},
                  # },
    "whole_mug": { 
                   "whole_target": target_marker | {"colorscale": "teal"},
                   "whole_source": source_marker | {"colorscale": "emrld"},
                   "whole_mug_pts": target_marker | {"colorscale": "reds"},
                 },
    "whole_rack": {
                   "whole_target": target_marker | {"colorscale": "teal"},
                   "whole_source": source_marker | {"colorscale": "emrld"},
                   "whole_tree_pts": target_marker | {"colorscale": "reds"},
                  },
    "no_rel_mug": {
                   "whole_target": target_marker | {"colorscale": "teal"},
                   "source_cup": source_marker | {"colorscale": "emrld"},
                   "source_handle": source_marker | {"colorscale": "purp"},
                   'source_cup_pts': source_marker | {"colorscale": "reds"}, 
                   'source_handle_pts': source_marker | {"colorscale": "reds"},
                  },
    "no_rel_rack": {
                    "whole_target": target_marker | {"colorscale": "teal"},
                    "source_trunk":  source_marker | {"colorscale": "emrld"},
                    "source_handle": source_marker | {"colorscale": "purp"},
                    'source_branch_pts': source_marker | {"colorscale": "reds"}, 
                    'source_trunk_pts': source_marker | {"colorscale": "reds"},
                    },
    "rel_mug": {
                "whole_target": target_marker | {"colorscale": "teal"},
                "source_cup_1":  source_marker | {"colorscale": "emrld"},
                "source_cup_0": source_marker | {"colorscale": "gray"},
                "source_handle_1": source_marker | {"colorscale": "purp"},
                "source_handle_0": source_marker | {"colorscale": "gray"}, 
                'source_cup_pts': source_marker | {"colorscale": "reds"}, 
                'source_handle_pts': source_marker | {"colorscale": "reds"},

                },
    "rel_rack": {
                "whole_target": target_marker | {"colorscale": "teal"},
                "source_trunk_1":  source_marker | {"colorscale": "emrld"},
                "source_trunk_0": source_marker | {"colorscale": "gray"},
                "source_branch_1": source_marker | {"colorscale": "purp"},
                "source_branch_0": source_marker | {"colorscale": "gray"}, 
                'source_branch_pts': source_marker | {"colorscale": "reds"}, 
                    'source_trunk_pts': source_marker | {"colorscale": "reds"}
                },
              }

names = ["target_mug", "whole_mug", "no_rel_mug", "rel_mug", "target_rack", "whole_rack", "no_rel_rack", "rel_rack"]

subplot_titles = ["Target Mug", "Whole Object Warping", "Part Warping", "With Relational Descriptors",
                  "Target Tree", "Whole Object Warping", "Part Warping", "With Relational Descriptors",]
viz_utils.show_pcd_grid_plotly(
    2,
    4,
    pcls=named_figs,
    markers=named_markers,
    names=names,
    subplot_titles=subplot_titles,
    # colorscales: Optional[List[str]] = None,
    # camera_views: Optional[List[Dict[str, dict]]] = None,
    # save_image: bool = False,
    # save_path: bool = False,
).show()