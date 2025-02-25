from rndf_robot.utils import util, path_util
import os, os.path as osp
import numpy as np
from src import utils, viz_utils
from src.utils import CanonPart, CanonPartMetadata
from scipy.spatial.transform import Rotation
from src.object_warping import (
    ObjectSE3Batch,
    ObjectWarpingSE3Batch,
    warp_to_pcd,
    warp_to_pcd_se3,
    warp_to_pcd_se3_hemisphere,
    PARAM_1,
    ALIGNMENT_PARAM,
    mask_and_cost_batch_pt,
)
import copy as cp
import pickle
from src import demo
from src.ndf_interface import NDFPartInterface

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-load_cached", action='store_true')
args = parser.parse_args()

demo_path = osp.join(path_util.get_rndf_data(), "relation_demos", "release_demos/mug_on_rack_relation")

parent_part_names = ['trunk', 'branch']
child_part_names = ['cup', 'handle']


parent_part_model_files = {'trunk': '/home/rthomp12/fewshot/part_based_warp_models/trunk_dict_20240412-042732', 
                           'branch': '/home/rthomp12/fewshot/part_based_warp_models/branch_dict_20240412-042732'}

child_part_model_files = {'cup': '/home/rthomp12/fewshot/part_based_warp_models/cup_dict_20240202-160637',
                          'handle': '/home/rthomp12/fewshot/part_based_warp_models/handle_dict_20240202-160637'}

parent_part_models = {part: CanonPart.from_pickle(parent_part_model_files[part]) for part in parent_part_names}
child_part_models = {part: CanonPart.from_pickle(child_part_model_files[part]) for part in child_part_names}


def check_segmentation_exists(pcl_id):
    root = "Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390"
    fn = os.path.join(root, pcl_id + ".txt")
    return os.path.exists(fn)

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

def load_demo(n):
    demo_files = [fn for fn in sorted(os.listdir(demo_path)) if fn.endswith(".npz")]
    demos = []
    demo = np.load(demo_path + "/" + demo_files[n], mmap_mode="r", allow_pickle=True)

    start_child_pcd = demo["multi_obj_start_pcd"].item()['child']
    final_child_pcd = demo["multi_obj_final_pcd"].item()['child']
    start_parent_pcd = demo["multi_obj_start_pcd"].item()['parent']
    final_parent_pcd = demo["multi_obj_final_pcd"].item()['parent']

    viz_utils.show_pcds_plotly({'child': demo["multi_obj_start_pcd"].item()['child'], 
                                'parent': utils.farthest_point_sample(demo["multi_obj_start_pcd"].item()['parent'], 200)[0]},
                                axis_visible=False).show()
    exit(0)

    start_pose_child = demo["multi_obj_start_obj_pose"].item()['child']
    start_pose_parent = demo["multi_obj_start_obj_pose"].item()['parent']
    final_pose_child = demo["multi_obj_final_obj_pose"].item()['child']
    final_pose_parent = demo["multi_obj_final_obj_pose"].item()['parent']

    demo_child_pcl_id =  demo["multi_object_ids"].item()['child']
    demo_parent_pcl_id = demo["multi_object_ids"].item()['parent']

    trans_child = utils.pos_quat_to_transform(
        start_pose_child[:3],
        start_pose_child[3:],
    )
    trans_parent = utils.pos_quat_to_transform(
        start_pose_parent[:3],
        start_pose_parent[3:],
    )

    child_demo_parts, _, child_demo_start_transforms = segment_mug(
        demo_child_pcl_id, trans_child
    )
               
    parent_demo_parts, _, parent_demo_start_transforms  = segment_mug_rack(
        start_parent_pcd#, trans_parent
    )

    return {
        'start_child_pcd': start_child_pcd, 
        'final_child_pcd': final_child_pcd,
        'start_parent_pcd': start_parent_pcd, 
        'final_parent_pcd': final_parent_pcd,
        'start_pose_child': start_pose_child,
        'final_pose_child': final_pose_child,
        'start_pose_parent': start_pose_parent, 
        'final_pose_parent': final_pose_parent,
        'child_demo_parts': child_demo_parts,
        'parent_demo_parts': parent_demo_parts,
    }

def reconstruct_objects(child_pcls, parent_pcls,):
    n_angles = 8

    PARAM_1 = {"lr": 1e-2, 
               "n_steps": 200,
               "n_samples": 1000, 
               "object_size_reg": 0.05} #.01

    inference_kwargs = {
                                "train_latents": True,
                                "train_scales": True,
                                "train_poses": True,
                            }

    mug_adjacent_part_pairs = [
    {"cup": child_pcls["cup"], "handle": child_pcls["handle"]},
    ]
    mug_canon_adjacent_part_pairs = [
        {"cup": child_part_models["cup"], "handle": child_part_models["handle"]},
    ]
    mug_ordered_part_pairs = [
        ("cup", "handle"),
    ]
    tree_adjacent_part_pairs = [
        {"trunk": parent_pcls["trunk"], "branch": parent_pcls["branch"]},
    ]
    tree_canon_adjacent_part_pairs = [
        {"trunk": parent_part_models["trunk"], "branch": parent_part_models["branch"]},
    ]
    tree_ordered_part_pairs = [
        ("trunk", "branch"),
    ]
    canon_parent_part_labels = {}
    canon_child_part_labels = {}

    def get_z_axis_labels(part_pcls, part_names):
        part_labels = {part: [] for part in part_names}
        for part in part_names: 
            z_mean = np.mean(part_pcls[part][:,2])
           
            part_labels[part].append(np.where(
                    part_pcls[part][:,2] > z_mean,
                    np.zeros_like(part_pcls[part][:, 0]),
                    np.ones_like(part_pcls[part][:, 0]),
                ))
        return part_labels

    parent_part_labels = {}
    child_part_labels = {}
    canon_parent_part_labels['relational'] = utils.get_canon_labels(
                    tree_canon_adjacent_part_pairs, parent_part_models, parent_part_names, 
                )
    
    canon_parent_part_labels['variational'] = get_z_axis_labels({part: parent_part_models[part].canonical_pcl for part in parent_part_names}, parent_part_names)

    parent_part_labels['relational'] = utils.get_part_labels(
        tree_adjacent_part_pairs,
    )
    parent_part_labels['variational'] = get_z_axis_labels(parent_pcls, parent_part_names)

    canon_child_part_labels['relational'] = utils.get_canon_labels(
                    mug_canon_adjacent_part_pairs, child_part_models, child_part_names, 
                )
    
    canon_child_part_labels['variational'] = get_z_axis_labels({part: child_part_models[part].canonical_pcl for part in child_part_names}, child_part_names)

    child_part_labels['relational'] = utils.get_part_labels(
        mug_adjacent_part_pairs,
    )

    child_part_labels['variational'] = get_z_axis_labels(child_pcls,child_part_names)
        
    child_reconstructions = {}
    child_params = {}
    parent_reconstructions = {}
    parent_params = {}

    device = 'cuda'


    for target_part in parent_part_names:
        canon = parent_part_models[target_part]
        canon_labels = {'relational': canon_parent_part_labels['relational'][target_part],
                        'variational': canon_parent_part_labels['variational'][target_part]}

        cost_function = (
            lambda source, target, canon_part_labels, latent_param, scale_param, initial_latents: mask_and_cost_batch_pt(
                target,
                parent_part_labels['relational'][target_part],
                source,
                canon_part_labels['relational'],
            ) + mask_and_cost_batch_pt(
                target,
                parent_part_labels['variational'][target_part],
                source,
                canon_part_labels['variational'],
            )
        )

        warp = ObjectWarpingSE3Batch(
                canon,
                parent_pcls[target_part],
                device,
                canon_labels=canon_labels,
                cost_function=cost_function,
                **cp.deepcopy(PARAM_1),
            )
        parent_reconstructions[target_part], _, parent_params[target_part] = warp_to_pcd_se3(
            warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
        )

    for target_part in child_part_names:
        print(target_part)
        canon = child_part_models[target_part]
        canon_labels = {'relational': canon_child_part_labels['relational'][target_part],
                        'variational': canon_child_part_labels['variational'][target_part]}

        cost_function = (
            lambda source, target, canon_part_labels, latent_param, scale_param, initial_latents: mask_and_cost_batch_pt(
                target,
                child_part_labels['relational'][target_part],
                source,
                canon_part_labels['relational'],
            ) + mask_and_cost_batch_pt(
                target,
                child_part_labels['variational'][target_part],
                source,
                canon_part_labels['variational'],
            )
        )
        
        warp = ObjectWarpingSE3Batch(
                canon,
                child_pcls[target_part],
                device,
                canon_labels=canon_labels,
                cost_function=cost_function,
                **cp.deepcopy(PARAM_1),
            )
        child_reconstructions[target_part], _, child_params[target_part] = warp_to_pcd_se3(
            warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
        )
    return child_reconstructions, parent_reconstructions, child_params, parent_params

def get_interaction_points(child_params, parent_params):
    (
        knns,
        deltas,
        target_indices,
    ) = demo.save_place_nearby_points_by_parts_v2(
        child_part_names,
        child_part_models,
        child_params,
        parent_part_names,
        parent_part_models,
        parent_params,
        0.03,
    )

    knn_pkl_file = 'temp_knn.pkl'
    pickle.dump({'knns': knns, 
              'deltas': deltas,
              'target_indices': target_indices}, open(knn_pkl_file, 'wb'))

    targets_child = {part: {} for part in child_part_names}
    targets_parent = {part: {} for part in child_part_names}

    for child_part in child_part_names:
        for parent_part in parent_part_names:
            if knns[child_part][parent_part] is None:
                continue
            anchors = child_part_models[child_part].to_pcd(child_params[child_part])[
                knns[child_part][parent_part]
            ]
            targets_child[child_part][parent_part] = np.mean(
                anchors + deltas[child_part][parent_part], axis=1
            )
            targets_parent[child_part][parent_part] = parent_part_models[
                parent_part
            ].to_pcd(parent_params[parent_part])[
                target_indices[child_part][parent_part]
            ] 

    child_part_targets = {}   
    child_targets_viz = {}
    canon_child_targets_viz = {}
    canon_parent_targets_viz = {}
    parent_targets_viz = {}

    for child_part in child_part_names:
        child_part_transform  = utils.pos_quat_to_transform(child_params[child_part].position, 
                                                      child_params[child_part].quat)
        for parent_part in parent_part_names:
            parent_part_transform  = utils.pos_quat_to_transform(parent_params[parent_part].position, 
                                                      parent_params[parent_part].quat)

            canon_child_targets_viz = canon_child_targets_viz | {f'child_targets_{child_part}_{parent_part}': targets_child[child_part][parent_part]}
            child_targets_viz =  child_targets_viz | {
                                 f'child_targets_{child_part}_{parent_part}': \
                                 utils.transform_pcd(targets_child[child_part][parent_part],
                                                     child_part_transform),
                                 }
            canon_parent_targets_viz = canon_parent_targets_viz | {f'parent_targets_{child_part}_{parent_part}': targets_parent[child_part][parent_part]}
            parent_targets_viz = parent_targets_viz | {
                                 f'parent_targets_{child_part}_{parent_part}': \
                                 utils.transform_pcd(targets_parent[child_part][parent_part],
                                                     parent_part_transform),
                                 }
    return child_targets_viz, parent_targets_viz, canon_child_targets_viz, canon_parent_targets_viz

def construct_constraint_pcl():
    knn_pkl_file = 'temp_knn.pkl'
    interaction_pt_data =  pickle.load(open(knn_pkl_file, 'rb'))
    knns = interaction_pt_data['knns']
    target_indices = interaction_pt_data['target_indices']
    deltas = interaction_pt_data['deltas']

    branch_raise = utils.pos_quat_to_transform([0,0,0], [0,0,0,1])#.15

    #assemble targets
    targets_child = {child_part: {} for child_part in child_part_names}
    targets_parent = {child_part: {} for child_part in child_part_names}
    for child_part in child_part_names:
        for parent_part in parent_part_names:
            if knns[child_part][parent_part] is None:
                targets_child[child_part][parent_part] = []
                targets_parent[child_part][parent_part] = []
                continue
            anchors = child_part_models[child_part].to_pcd(child_params[child_part])[
                knns[child_part][parent_part]
            ]
            targets_child[child_part][parent_part] = np.mean(
                anchors + deltas[child_part][parent_part], axis=1
            ) 

            targets_parent[child_part][parent_part] = parent_part_models[
                parent_part
            ].to_pcd(parent_params[parent_part])[
                target_indices[child_part][parent_part]
            ] 

    targets_parent['cup']['branch'] = utils.transform_pcd(targets_parent['cup']['branch'], branch_raise)
    targets_parent['handle']['branch'] = utils.transform_pcd(targets_parent['handle']['branch'], branch_raise)

    #concatenate targets on the branch and trunk
    concat_targets_child = np.concatenate([targets_child[child_part][parent_part] for parent_part in parent_part_names for child_part in child_part_names])
    concat_targets_parent = np.concatenate([targets_parent[child_part][parent_part] for parent_part in parent_part_names for child_part in child_part_names])
    #use to get final constraint pose 
    whole_poses = utils.best_fit_transform(concat_targets_child, concat_targets_parent)[0]#{child_part: utils.best_fit_transform(concat_targets_child[child_part], concat_targets_parent[child_part])[0] for child_part in child_part_names}

    fail_poses = {'cup': utils.best_fit_transform(targets_child['cup']['trunk'], targets_parent['cup']['trunk'])[0], 'handle': utils.best_fit_transform(targets_child['handle']['branch'], targets_parent['handle']['branch'])[0]}

    success_poses = {'cup': utils.best_fit_transform(targets_child['cup']['branch'], targets_parent['cup']['branch'])[0], 'handle': utils.best_fit_transform(targets_child['handle']['branch'], targets_parent['handle']['branch'])[0]}


    child_targets_viz = {}
    canon_child_targets_viz = {}
    canon_parent_targets_viz = {}
    parent_targets_viz = {}

    for child_part in child_part_names:
        child_part_transform  = utils.pos_quat_to_transform(child_params[child_part].position, 
                                                      child_params[child_part].quat)
        for parent_part in parent_part_names:
            parent_part_transform  = utils.pos_quat_to_transform(parent_params[parent_part].position, 
                                                      parent_params[parent_part].quat)

            canon_child_targets_viz = canon_child_targets_viz | {f'child_targets_{child_part}_{parent_part}': targets_child[child_part][parent_part]}
            # child_targets_viz =  child_targets_viz | {
            #                      f'child_targets_{child_part}_{parent_part}': \
            #                      utils.transform_pcd(targets_child[child_part][parent_part],
            #                                          child_part_transform),
            #                      }
            canon_parent_targets_viz = canon_parent_targets_viz | {f'parent_targets_{child_part}_{parent_part}': targets_parent[child_part][parent_part]}
            # parent_targets_viz = parent_targets_viz | {
            #                      f'parent_targets_{child_part}_{parent_part}': \
            #                      utils.transform_pcd(targets_parent[child_part][parent_part],
            #                                          parent_part_transform),
            #                      }

    demo_interaction_points_camera = dict(
                            up=dict(x=0, y=0, z=1),
                            center = dict(x=0, y=0, z=0.05),
                            eye=dict(x=-1.7,y=-.9,z=0.2))
    
    pcd_marker = {
            "size": 5,
            "opacity": 1
        }
    ip_marker = {
        "size": 10,
        "opacity": 1
    }

    markers = {'mug': pcd_marker,
              'cup': pcd_marker,
              'handle': pcd_marker,
              'rack': pcd_marker,
              'branch': pcd_marker,
              'trunk': pcd_marker,
              'cup_i': ip_marker,
              'mug_i': ip_marker,
              'handle_i': ip_marker,
              'branch_i': ip_marker,
              'rack_i': ip_marker,
              'trunk_i': ip_marker,}
    
    colors= {'mug': 'magenta',
              'cup': 'magenta',
              'handle': 'magenta',
              'rack': 'blues',
              'branch': 'blues',
              'trunk': 'blues',
              'cup_i': 'oranges',
              'mug_i': 'oranges',
              'handle_i': 'oranges',
              'branch_i': 'greens',
              'rack_i': 'greens',
              'trunk_i': 'greens',}

    trunk_model = np.concatenate([parent_part_models['trunk'].canonical_pcl, 
                                  utils.transform_pcd(parent_part_models['trunk'].canonical_pcl, utils.pos_quat_to_transform([0,0,.15], [0,0,0,1]))])
    
    viz_utils.show_pcds_plotly({'mug': np.concatenate([child_part_models['cup'].canonical_pcl,  child_part_models['handle'].canonical_pcl]),
                                'rack': np.concatenate([parent_part_models['trunk'].canonical_pcl , parent_part_models['branch'].canonical_pcl])}
                                | {'mug_i': concat_targets_child, 'rack_i': concat_targets_parent, }
                                ,
                                center=True, 
                                axis_visible=False,
                                #show_legend=False,
                                markers= markers,
                                #colors=colors,
                                camera=demo_interaction_points_camera).show()
    
    viz_utils.show_pcds_plotly({'cup': utils.transform_pcd(child_part_models['cup'].canonical_pcl, fail_poses['cup']), 
                                'handle':  utils.transform_pcd(child_part_models['handle'].canonical_pcl, fail_poses['handle']),
                                'trunk': trunk_model, 
                                'branch': utils.transform_pcd(parent_part_models['branch'].canonical_pcl,branch_raise)}
                                | {'cup_i': utils.transform_pcd(targets_child['cup']['trunk'] + (np.random.uniform(targets_child['cup']['trunk'])*.005), fail_poses['cup']), 
                                                 'trunk_i': targets_parent['cup']['trunk'] + (np.random.uniform(targets_child['cup']['trunk'])*.001), 
                                                 'handle_i': utils.transform_pcd(targets_child['handle']['branch'], fail_poses['handle']),
                                                 'branch_i': targets_parent['handle']['branch'],},
                                center=True, 
                                axis_visible=False,
                                show_legend=False,
                                markers= markers,
                                colors=colors,
                                camera=demo_interaction_points_camera).show()
    
    viz_utils.show_pcds_plotly({'cup': utils.transform_pcd(child_part_models['cup'].canonical_pcl, success_poses['cup']), 
                                'handle':  utils.transform_pcd(child_part_models['handle'].canonical_pcl, success_poses['handle']),
                                'trunk': trunk_model, 
                                'branch': utils.transform_pcd(parent_part_models['branch'].canonical_pcl,branch_raise)}
                                | {'cup_i': utils.transform_pcd(targets_child['cup']['branch'], success_poses['cup']) +(np.random.uniform(targets_child['cup']['trunk'])*.001), 
                                                 'trunk_i': targets_parent['cup']['branch'] + (np.random.uniform(targets_child['cup']['trunk'])*.001), 
                                                 'handle_i': utils.transform_pcd(targets_child['handle']['branch'], success_poses['handle']),
                                                 'branch_i': targets_parent['handle']['branch'],},
                                center=True, 
                                axis_visible=False,
                                show_legend=False,
                                markers= markers,
                                colors=colors,
                                camera=demo_interaction_points_camera).show()

def transfer_by_program(constraint_pair_program, child_pcds, parent_pcds,):
    
    knn_pkl_file = 'temp_knn.pkl'

    interface = NDFPartInterface(
        canon_source_parts_paths = child_part_model_files,
        canon_target_parts_paths = parent_part_model_files,
        source_part_names = ["cup", "handle"],
        target_part_names = ["trunk", "branch"],
        )

    final_transform, part_transforms = interface.infer_relpose(
        demo_data['child_demo_parts'],
        demo_data['parent_demo_parts'],
        constraint_pair_program,
        se3 = True,
        knn_pkl=knn_pkl_file,
        return_part_transforms = True
    )
        # if knn_pkl is not None:
        #     demo_dict = pickle.load(open(knn_pkl, "rb"))
        #     self.knns = demo_dict["knns"]
        #     self.deltas = demo_dict["deltas"]
        #     self.target_indices = demo_dict["target_indices"]

    return final_transform, part_transforms

demo_data = load_demo(4)

demo_start_fig_camera = dict(
                            up=dict(x=0, y=0, z=1),
                            center = dict(x=0, y=0, z=0.05),
                            eye=dict(x=-1.7,y=-.9,z=0.2))
# demo_start_fig = viz_utils.show_pcds_plotly({'mug': demo_data['start_child_pcd'], 
#                                              'rack': demo_data['start_parent_pcd']}, 
#                                              center=True,
#                                              axis_visible=False,
#                                              show_legend=False,
#                                              camera=demo_start_fig_camera)

# demo_start_fig.show()

# demo_end_fig_camera = dict(
#                             up=dict(x=0, y=0, z=1),
#                             center = dict(x=0, y=0, z=0.05),
#                             eye=dict(x=-1.7,y=-.9,z=0.2))
# demo_end_fig = viz_utils.show_pcds_plotly({'mug': demo_data['final_child_pcd'], 
#                                            'rack': demo_data['final_parent_pcd']}, 
#                                            center=True, 
#                                            axis_visible=False,
#                                            show_legend=False,
#                                            camera=demo_end_fig_camera)
# demo_end_fig.show()



final_cup_transform = np.matmul(utils.pos_quat_to_transform(demo_data['final_pose_child'][:3], 
                                                            demo_data['final_pose_child'][3:]),
                                np.linalg.inv(utils.pos_quat_to_transform(
                                                            demo_data['start_pose_child'][:3], 
                                                            demo_data['start_pose_child'][3:])))

for part in demo_data['child_demo_parts'].keys():
    demo_data['child_demo_parts'][part] = utils.transform_pcd(demo_data['child_demo_parts'][part] , final_cup_transform)

demo_segmented_camera = dict(
                            up=dict(x=0, y=0, z=1),
                            center = dict(x=0, y=0, z=0.05),
                            eye=dict(x=-1.7,y=-.9,z=0.2))



demo_segmented = viz_utils.show_pcds_plotly({'cup': demo_data['child_demo_parts']['cup'], 
                            'handle': demo_data['child_demo_parts']['handle'], 
                            'trunk': demo_data['parent_demo_parts']['trunk'],
                            'branch': demo_data['parent_demo_parts']['branch'], }, 
                            center=True, 
                            axis_visible=False,
                            show_legend=False,
                            camera=demo_segmented_camera)
demo_segmented.update_layout(
        title=dict(text="Part Segmentation",  x=0.5),
    )
demo_segmented.show()

if not args.load_cached: 
    child_reconstructions, parent_reconstructions, child_params, parent_params = reconstruct_objects(demo_data['child_demo_parts'], demo_data['parent_demo_parts'])

    viz_utils.show_pcds_plotly(child_reconstructions|parent_reconstructions, camera=demo_segmented_camera).show()
    child_targets_viz, parent_targets_viz, canon_child_targets_viz, canon_parent_targets_viz = get_interaction_points(child_params, parent_params)
    
    cache = {}
    cache['child_reconstructions'] = child_reconstructions 
    cache['parent_reconstructions'] = parent_reconstructions 
    cache['child_params'] = child_params
    cache['parent_params'] = parent_params
    cache['child_targets_viz'] = child_targets_viz
    cache['parent_targets_viz'] = parent_targets_viz
    cache['canon_child_targets_viz'] = canon_child_targets_viz
    cache['canon_parent_targets_viz'] = canon_parent_targets_viz
    pickle.dump(cache, open('big_fig_cache.txt', 'wb'))
else:
    cache = pickle.load(open('big_fig_cache.txt', 'rb'))
    child_reconstructions = cache['child_reconstructions']
    parent_reconstructions = cache['parent_reconstructions']
    child_params = cache['child_params']
    parent_params = cache['parent_params']
    child_targets_viz = cache['child_targets_viz']
    parent_targets_viz = cache['parent_targets_viz']
    canon_child_targets_viz = cache['canon_child_targets_viz']
    canon_parent_targets_viz = cache['canon_parent_targets_viz']


demo_interaction_points = viz_utils.show_pcds_plotly({'reconstructed_cup': child_reconstructions['cup'], 
                                                      'reconstructed_handle': child_reconstructions['handle'], 
                                                      'reconstructed_trunk': parent_reconstructions['trunk'],
                                                      'reconstructed_branch': parent_reconstructions['branch'],
                                                      } | child_targets_viz | parent_targets_viz | {'mug': demo_data['final_child_pcd']} , 
                                                      center=True, 
                                                      axis_visible=False).show()

                                                    
# demo_interaction_points.update_layout(
#         title=dict(text="Identifying Interaction Points",  x=0.5),
#     )

# # on the rack but reconstruction + interaction points

# canon_segmented_child = viz_utils.show_pcds_plotly({'canon_cup': child_part_models['cup'].canonical_pcl , 
#                             'canon_handle': child_part_models['handle'].canonical_pcl,} | canon_child_targets_viz, center=True, axis_visible=False).show()
                            
# canon_segmented_parent = viz_utils.show_pcds_plotly({'canon_trunk': parent_part_models['trunk'].canonical_pcl , 
#                             'canon_branch': parent_part_models['branch'].canonical_pcl,} | canon_parent_targets_viz, center=True, axis_visible=False).show()


# #TODO - Why isn't the Final Transform working??
# demo_final_transform, demo_part_transforms = transfer_by_program([('cup', 'trunk'), ('handle', 'branch')], demo_data['child_demo_parts'], demo_data['parent_demo_parts'])
# constraint_pcds = {}
# for pair in [('cup', 'trunk'), ('handle', 'branch')]:
#     constraint_pcds[pair[0]] = utils.transform_pcd(demo_data['child_demo_parts'][pair[0]], demo_part_transforms[pair[0]][pair[1]])

# transformed_final = utils.transform_pcd(demo_data['start_child_pcd'], demo_final_transform) 

# pair_program_fail = viz_utils.show_pcds_plotly(constraint_pcds | {'reconstructed_trunk': parent_reconstructions['trunk'],
#                                                                   'reconstructed_branch': parent_reconstructions['branch']} \
#                                                                |{'final': transformed_final},)
# pair_program_fail.show()

# exit(0)
# cache['failure_constraint_pcds'] = constraint_pcds
# cache['failure_final_transform'] = demo_final_transform

# demo_final_transform, demo_part_transforms = transfer_by_program([('cup', 'branch'), ('handle', 'branch')], demo_data['child_demo_parts'], demo_data['parent_demo_parts'])
# constraint_pcds = {}
# for pair in [('cup', 'branch'), ('handle', 'branch')]:
#     constraint_pcds[pair[0]] = utils.transform_pcd(demo_data['child_demo_parts'][pair[0]], demo_part_transforms[pair[0]][pair[1]])

# transformed_final = utils.transform_pcd(demo_data['start_child_pcd'], demo_final_transform) 

# pair_program_success = viz_utils.show_pcds_plotly(constraint_pcds | {'reconstructed_trunk': parent_reconstructions['trunk'],
#                                                                   'reconstructed_branch': parent_reconstructions['branch']} \
#                                                                |{'final': transformed_final} | {'start': demo_data['start_child_pcd']})
# pair_program_success.show()


# cache['success_constraint_pcds'] = constraint_pcds
# cache['success_final_transform'] = demo_final_transform

# pickle.dump(cache, open('big_fig_cache.txt', 'wb'))
#TODO debug w child demo parts and etc
# exit(0)

#pair program success
#and need to save all the images



construct_constraint_pcl()


# # test_setting_initial_state
test_data = load_demo(6)
viz_utils.show_pcds_plotly({'mug': test_data['start_child_pcd'], 
                            'rack': test_data['start_parent_pcd']},center=True, 
                            axis_visible=False).show()

# for part in test_data['child_demo_parts'].keys():
#     test_data['child_demo_parts'][part] = utils.transform_pcd(test_data['child_demo_parts'][part] , final_cup_transform)

test_segmented = viz_utils.show_pcds_plotly({'cup': test_data['child_demo_parts']['cup'], 
                            'handle': test_data['child_demo_parts']['handle'], 
                            'trunk': test_data['parent_demo_parts']['trunk'],
                            'branch': test_data['parent_demo_parts']['branch'], }, 
                            center=True, 
                            axis_visible=False)
# test_segmented.show()


(test_child_reconstructions, 
 test_parent_reconstructions, 
 test_child_params, 
 test_parent_params) = reconstruct_objects(test_data['child_demo_parts'], test_data['parent_demo_parts'])

# (test_child_targets_viz, 
#  test_parent_targets_viz, 
#  test_canon_child_targets_viz, 
#  test_canon_parent_targets_viz) = get_interaction_points(child_params, parent_params)

demo_interaction_points = viz_utils.show_pcds_plotly({'reconstructed_cup': test_child_reconstructions['cup'], 
                                                      'reconstructed_handle': test_child_reconstructions['handle'], 
                                                      'reconstructed_trunk': test_parent_reconstructions['trunk'],
                                                      'reconstructed_branch': test_parent_reconstructions['branch'],
                                                      }, #| test_child_targets_viz | test_parent_targets_viz, 
                                                      center=True, 
                                                      axis_visible=False).show()

#infer relpose next and pull final
#load knns 
knn_pkl_file = 'temp_knn.pkl'
interaction_pt_data =   pickle.load(open(knn_pkl_file, 'rb'))
knns = interaction_pt_data['knns']
target_indices = interaction_pt_data['target_indices']
deltas = interaction_pt_data['deltas']

test_targets_child = {part: {} for part in child_part_names}
test_targets_parent = {part: {} for part in child_part_names}

for child_part in child_part_names:
    for parent_part in parent_part_names:
        if knns[child_part][parent_part] is None:
            continue
        anchors = child_part_models[child_part].to_pcd(test_child_params[child_part])[
            knns[child_part][parent_part]
        ]
        test_targets_child[child_part][parent_part] = np.mean(
            anchors + deltas[child_part][parent_part], axis=1
        )
        test_targets_parent[child_part][parent_part] = parent_part_models[
            parent_part
        ].to_pcd(test_parent_params[parent_part])[
            target_indices[child_part][parent_part]
        ] 


 
test_child_targets_viz = {}
test_parent_targets_viz = {}

for child_part in child_part_names:
    child_part_transform  = utils.pos_quat_to_transform(test_child_params[child_part].position, 
                                                  test_child_params[child_part].quat)
    for parent_part in parent_part_names:
        parent_part_transform  = utils.pos_quat_to_transform(test_parent_params[parent_part].position, 
                                                  test_parent_params[parent_part].quat)
        test_child_targets_viz =  test_child_targets_viz | {
                             f'child_targets_{child_part}_{parent_part}': \
                             utils.transform_pcd(test_targets_child[child_part][parent_part],
                                                 child_part_transform),
                             }
        test_parent_targets_viz = test_parent_targets_viz | {
                             f'parent_targets_{child_part}_{parent_part}': \
                             utils.transform_pcd(test_targets_parent[child_part][parent_part],
                                                 parent_part_transform),
                             }


interface = NDFPartInterface(
        canon_source_parts_paths = child_part_model_files,
        canon_target_parts_paths = parent_part_model_files,
        source_part_names = child_part_names,
        target_part_names = parent_part_names,)


trans_predicted, constraint_pcl = interface.infer_relpose(
        test_data['child_demo_parts'], 
        test_data['parent_demo_parts'], 
        [('cup', 'branch'),('handle', 'branch')], 
        se3=True, 
        child_params=test_child_params, 
        parent_params=test_parent_params, 
        knn_pkl=knn_pkl_file, 
        return_constraint_pcl=True
    )


pcd_marker = {
            "size": 5,
            "opacity": 1
        }
ip_marker = {
    "size": 10,
    "opacity": .35
}

markers = {'mug': pcd_marker,
           
            'rack': pcd_marker,
            'constraint': ip_marker,}


colors= {'constraint': 'magenta',
            'mug': 'Plotly3',
            'rack': 'Viridis',}


demo_interaction_points_camera = dict(
                        up=dict(x=0, y=0, z=1),
                        center = dict(x=0, y=0, z=0.05),
                        eye=dict(x=-1.7,y=-.9,z=0.2))
    

viz_utils.show_pcds_plotly({'constraint': constraint_pcl, 
                'mug': utils.transform_pcd(test_data['start_child_pcd'], trans_predicted),
                'rack': test_data['start_parent_pcd']}, camera=demo_interaction_points_camera , markers=markers,  colors=colors, axis_visible=False).show()


exit(0)

# pcd_marker = {
#             "size": 5,
#             "opacity": .0075
#         }
# ip_marker = {
#     "size": 10,
#     "opacity": 1
# }

# markers = {key: pcd_marker for key in test_child_reconstructions.keys()}| \
#           {key: pcd_marker for key in test_parent_reconstructions.keys()}| \
# {key: ip_marker for key in test_child_targets_viz.keys()}| \
# {key: ip_marker for key in test_parent_targets_viz.keys()} 


# viz_utils.show_pcds_plotly(test_child_reconstructions|test_parent_reconstructions|
#                             test_child_targets_viz|test_parent_targets_viz, camera=demo_segmented_camera,
#                               markers=markers, axis_visible=False).show()

# exit(0)




# demo_interaction_points = viz_utils.show_pcds_plotly({'reconstructed_cup': test_child_reconstructions['cup'], 
#                                                       'reconstructed_handle': test_child_reconstructions['handle'], 
#                                                       'reconstructed_trunk': test_parent_reconstructions['trunk'],
#                                                       'reconstructed_branch': test_parent_reconstructions['branch'],
#                                                       }| test_child_targets_viz | test_parent_targets_viz, 
#                                                       center=True, 
#                                                       axis_visible=False).show()



# knn_pkl_file = 'temp_knn.pkl'
# interaction_pt_data =  pickle.load(open(knn_pkl_file, 'rb'))
# knns = interaction_pt_data['knns']
# target_indices = interaction_pt_data['target_indices']
# deltas = interaction_pt_data['deltas']

# branch_raise = utils.pos_quat_to_transform([0,0,.15], [0,0,0,1])

# #assemble targets
# targets_child = {child_part: {} for child_part in child_part_names}
# targets_parent = {child_part: {} for child_part in child_part_names}
# for child_part in child_part_names:
#     for parent_part in parent_part_names:
#         if knns[child_part][parent_part] is None:
#             targets_child[child_part][parent_part] = []
#             targets_parent[child_part][parent_part] = []
#             continue
#         anchors = child_part_models[child_part].to_pcd(child_params[child_part])[
#             knns[child_part][parent_part]
#         ]
#         targets_child[child_part][parent_part] = np.mean(
#             anchors + deltas[child_part][parent_part], axis=1
#         ) 

#         targets_parent[child_part][parent_part] = parent_part_models[
#             parent_part
#         ].to_pcd(parent_params[parent_part])[
#             target_indices[child_part][parent_part]
#         ] 

# success_poses = {'cup': utils.best_fit_transform(targets_child['cup']['branch'], targets_parent['cup']['branch'])[0], 'handle': utils.best_fit_transform(targets_child['handle']['branch'], targets_parent['handle']['branch'])[0]}


# #get the full pose transfer


# child_targets_viz = {}
# canon_child_targets_viz = {}
# canon_parent_targets_viz = {}
# parent_targets_viz = {}
# trans_cs_to_ct = {part: {} for part in child_part_names}

# for child_part in child_part_names:
#     child_part_transform  = utils.pos_quat_to_transform(child_params[child_part].position, 
#                                                     child_params[child_part].quat)
#     for parent_part in parent_part_names:
#         parent_part_transform  = utils.pos_quat_to_transform(parent_params[parent_part].position, 
#                                                     parent_params[parent_part].quat)

#         # canon_child_targets_viz = canon_child_targets_viz | {f'child_targets_{child_part}_{parent_part}': targets_child[child_part][parent_part]}
#         child_targets_viz =  child_targets_viz | {
#                              f'child_targets_{child_part}_{parent_part}': \
#                              utils.transform_pcd(targets_child[child_part][parent_part],
#                                                  child_part_transform),
#                              }
#         # canon_parent_targets_viz = canon_parent_targets_viz | {f'parent_targets_{child_part}_{parent_part}': targets_parent[child_part][parent_part]}
#         parent_targets_viz = parent_targets_viz | {
#                              f'parent_targets_{child_part}_{parent_part}': \
#                              utils.transform_pcd(targets_parent[child_part][parent_part],
#                                                  parent_part_transform),
#                              }

#         trans_cs_to_ct[child_part][parent_part], _, _ = utils.best_fit_transform(
#                             np.array(targets_child[child_part][parent_part]).squeeze(),
#                             np.array(targets_parent[child_part][parent_part]).squeeze(),
#                         )
        

# pcd_marker = {
#             "size": 5,
#             "opacity": 1
#         }
# ip_marker = {
#     "size": 10,
#     "opacity": 1
# }

# markers = {'mug': pcd_marker,
#             'cup': pcd_marker,
#             'handle': pcd_marker,
#             'rack': pcd_marker,
#             'branch': pcd_marker,
#             'trunk': pcd_marker,
#             'cup_i': ip_marker,
#             'mug_i': ip_marker,
#             'handle_i': ip_marker,
#             'branch_i': ip_marker,
#             'rack_i': ip_marker,
#             'trunk_i': ip_marker,}


# colors= {'mug': 'magenta',
#             'cup': 'Plotly3',
#             'handle': 'Viridis',
#             'rack': 'blues',
#             'branch': 'greens',
#             'trunk': 'blues',
#             'cup_i': 'greens',
#             'mug_i': 'greens',
#             'handle_i': 'greens',
#             'branch_i': 'oranges',
#             'rack_i': 'oranges',
#             'trunk_i': 'oranges',}


# demo_interaction_points_camera = dict(
#                         up=dict(x=0, y=0, z=1),
#                         center = dict(x=0, y=0, z=0.05),
#                         eye=dict(x=-1.7,y=-.9,z=0.2))
    


# viz_utils.show_pcds_plotly({'cup': test_child_reconstructions['cup'],
#                             'handle': test_child_reconstructions['handle'],
#                             'trunk': test_parent_reconstructions['trunk'], 
#                             'branch': test_parent_reconstructions['branch']}
#                             | {'cup_i': utils.transform_pcd(targets_child['cup']['branch'], utils.pos_quat_to_transform(test_child_params['cup'].position, test_child_params['cup'].quat)), 
#                                'handle_i': utils.transform_pcd(targets_child['handle']['branch'],   utils.pos_quat_to_transform(test_child_params['handle'].position, test_child_params['handle'].quat)),
#                                'trunk_i':utils.transform_pcd(targets_parent['cup']['branch'],utils.pos_quat_to_transform(test_parent_params['branch'].position, test_parent_params['branch'].quat)),
#                                'branch_i':  utils.transform_pcd(targets_parent['handle']['branch'], utils.pos_quat_to_transform(test_parent_params['branch'].position, test_parent_params['branch'].quat)),}
#                             ,
#                             center=True, 
#                             axis_visible=False,
#                             show_legend=False,
#                             markers= markers,
#                             colors=colors,
#                             camera=demo_interaction_points_camera).show()




# trans_s_to_b = {}
# for part in child_part_names:
#     trans_s_to_b[part] = utils.pos_quat_to_transform(
#        test_child_params[part].position, test_child_params[part].quat
#     )

# trans_t_to_b = {}
# for part in parent_part_names:
#     trans_t_to_b[part] = utils.pos_quat_to_transform(
#         test_parent_params[part].position, test_parent_params[part].quat
#     )

# trans_s_to_t = {part: {} for part in child_part_names}
# for part in child_part_names:
#     for target_part in parent_part_names:
#         if knns[part][target_part] is None:
#             continue
#         trans_s_to_t[part][target_part] = (
#             trans_t_to_b[target_part]
#             @ trans_cs_to_ct[part][target_part]
#             @ np.linalg.inv(trans_s_to_b[part]))


# pcd_marker = {
#             "size": 5,
#             "opacity": 1
#         }
# ip_marker = {
#     "size": 10,
#     "opacity": 1
# }

# markers = {'mug': pcd_marker,
#             'cup': pcd_marker,
#             'handle': pcd_marker,
#             'rack': pcd_marker,
#             'branch': pcd_marker,
#             'trunk': pcd_marker,
#             'cup_i': ip_marker,
#             'mug_i': ip_marker,
#             'handle_i': ip_marker,
#             'branch_i': ip_marker,
#             'rack_i': ip_marker,
#             'trunk_i': ip_marker,}


# colors= {'mug': 'magenta',
#             'cup': 'magenta',
#             'handle': 'magenta',
#             'rack': 'blues',
#             'branch': 'blues',
#             'trunk': 'blues',
#             'cup_i': 'oranges',
#             'mug_i': 'oranges',
#             'handle_i': 'oranges',
#             'branch_i': 'greens',
#             'rack_i': 'greens',
#             'trunk_i': 'greens',}


# demo_interaction_points_camera = dict(
#                         up=dict(x=0, y=0, z=1),
#                         center = dict(x=0, y=0, z=0.05),
#                         eye=dict(x=-1.7,y=-.9,z=0.2))
    

# viz_utils.show_pcds_plotly({'cup': utils.transform_pcd(test_child_reconstructions['cup'], trans_s_to_t['cup']['branch']), 
#                             'handle':  utils.transform_pcd(test_child_reconstructions['handle'],trans_s_to_t['handle']['branch']),
#                             'trunk': test_parent_reconstructions['trunk'], 
#                             'branch': test_parent_reconstructions['branch']}
#                             | {'cup_i': utils.transform_pcd(targets_child['cup']['branch'], np.matmul(trans_s_to_t['cup']['branch'], utils.pos_quat_to_transform(test_child_params['cup'].position, test_child_params['cup'].quat))), 
#                                'handle_i': utils.transform_pcd(targets_child['handle']['branch'], np.matmul(trans_s_to_t['handle']['branch'],  utils.pos_quat_to_transform(test_child_params['handle'].position, test_child_params['handle'].quat))),
#                                'trunk_i':utils.transform_pcd(targets_parent['cup']['branch'],utils.pos_quat_to_transform(test_parent_params['branch'].position, test_parent_params['branch'].quat)),
#                                'branch_i':  utils.transform_pcd(targets_parent['handle']['branch'], utils.pos_quat_to_transform(test_parent_params['branch'].position, test_parent_params['branch'].quat)),}
#                             ,
#                             center=True, 
#                             axis_visible=False,
#                             show_legend=False,
#                             markers= markers,
#                             colors=colors,
#                             camera=demo_interaction_points_camera).show()