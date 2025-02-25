import numpy as np
from src import viz_utils
from src import utils
import importlib
from src.utils import CanonPart, CanonPartMetadata, get_pointcloud_in_cam_frame, transform_cloud_to_base, remove_outliers
from PIL import Image
import torch
import open3d as o3d
import open3d.visualization.gui as gui
import copy as cp
import pickle
import matplotlib
from sklearn.decomposition import PCA
from scripts.mug_utils import *

from src.object_warping import (
    ObjectWarpingSE2Batch,
    ObjectSE2Batch,
    ObjectSE3Batch,
    ObjectWarpingSE3Batch,
    warp_to_pcd,
    warp_to_pcd_se2,
    warp_to_pcd_se3,
    warp_to_pcd_se3_hemisphere,
    PARAM_1,
    ALIGNMENT_PARAM,
    mask_and_cost_batch_pt,
)


#TODO: move size reg, add visualization of the descriptors for debugging purposes
#Currently looking like scale regularization is gonna struggle, at least with the bad pointclouds/chamfer distance issue
#So an additional descriptor seems like a better bet than the pca remapping situation. look at the variance alignment thing
#'soft' descriptors might also be a good option, maybe try and implement that
#save some configs to demonstrate things to ondrej/include in a blog post or whatever


#load the segmented pcl
save_name = '/home/rthomp12/fewshot/ridged_mug_flared_rack_grasp/init_scene_pcls.npz'
scene_pcls = np.load(save_name)
masked_pcls = {k: scene_pcls[k] for k in scene_pcls.keys()}

child_part_names = ['cup', 'handle']

def load_mug_parts(idx):
    training_ids = load_all_shapenet_files("mug")
    test_id = training_ids[idx]

    mug_mesh = get_mesh(test_id)
    part_meshes = get_segmented_mesh(test_id)

    mug_part_pcls = {}

    for part_id in part_meshes.keys():
        mug_part_pcls[child_part_names[part_id-37]] = utils.trimesh_create_verts_surface(part_meshes[part_id], 1000)

    mug_part_pcls['cup'],  mug_part_pcls['handle'] = utils.scale_points_circle([mug_part_pcls['cup'],  mug_part_pcls['handle']], base_scale=.1)

    return mug_part_pcls

masked_pcls = load_mug_parts(4)
target_part = 'handle'


child_part_model_files = {'cup': '/home/rthomp12/fewshot/part_based_warp_models/cup_dict_20240202-160637',
                          'handle': '/home/rthomp12/fewshot/part_based_warp_models/handle_dict_20240202-160637'}


child_part_models = {part: CanonPart.from_pickle(child_part_model_files[part]) for part in child_part_names}

child_adjacent_part_pairs = [
    {"cup": masked_pcls["cup"], "handle": masked_pcls["handle"]},
]
child_canon_adjacent_part_pairs = [
    {"cup": child_part_models["cup"], "handle": child_part_models["handle"]},
]
child_ordered_part_pairs = [
    ("cup", "handle"),
]

canon_child_part_labels = utils.get_canon_labels(
                child_canon_adjacent_part_pairs, 
                child_part_models, 
                child_part_names, 
            )

child_part_labels = utils.get_part_labels(
    child_adjacent_part_pairs,
)

# print(child_part_labels)
# exit(0)



def get_pca_descriptors(part_pcls, part_names):
    n_descriptors = 1
    part_pcas = {}
    part_labels = {part: [] for part in part_names}
    for part in part_names: 
        part_pcas[part] = PCA(n_components=2)
        components = part_pcas[part].fit_transform(part_pcls[part]).T
        for i in range(n_descriptors):
            component_mean = np.mean(components[i, :])
            part_labels[part].append(np.where(
                    components[i,:] > component_mean,
                    np.zeros_like(part_pcls[part][:, 0]),
                    np.ones_like(part_pcls[part][:, 0]),
                ))
    return part_labels

def display_descriptors(part_pcls, part_labels, part_names, root='', colors=None):
    marker={
            "size": 5,
            "opacity": 1
        }
    descriptor_pcls = {}
    markers = {}

        
    for part in part_names:
        if colors is not None:
            assert len(colors) == len(part_labels[part])
        for i in range(len(part_labels[part])):
            descriptor_pcls[f'{root}_{part}_{i}_0'] = part_pcls[part][part_labels[part][i] == 0]
            descriptor_pcls[f'{root}_{part}_{i}_1'] = part_pcls[part][part_labels[part][i] == 1]
            if colors is not None:
                markers[f'{root}_{part}_{i}_0'] = marker | {'colorscale': colors[i][0]}
                markers[f'{root}_{part}_{i}_1'] = marker | {'colorscale': colors[i][1]}
            else:
                markers[f'{root}_{part}_{i}_0'] = marker 
                markers[f'{root}_{part}_{i}_1'] = marker 

    return descriptor_pcls, markers


# part_labels = get_pca_descriptors(masked_pcls, ['cup', 'handle'])
# descriptors_pcls = display_descriptors(masked_pcls, part_labels, ['cup', 'handle'])
# viz_utils.show_pcds_plotly(descriptors_pcls).show()

# exit(0)

def warp_with_objective(cost_function,):
    device = 'cuda'
    n_angles = 8

    PARAM_1 = {"lr": 1e-2, 
               "n_steps": 200,
               "n_samples": 1000, 
               "object_size_reg": 0.0} #.01

    inference_kwargs = {
                            "train_latents": True,
                            "train_scales": True,
                            "train_poses": True,
                            }

    pca_descriptor = get_pca_descriptors({target_part: child_part_models[target_part].canonical_pcl}, [target_part])[target_part]

    canon_labels = {'relational': canon_child_part_labels[target_part], 'variational': pca_descriptor}

    if cost_function is None:
        warp = ObjectWarpingSE3Batch(
        child_part_models[target_part],
        masked_pcls[target_part],
        device,
        **cp.deepcopy(PARAM_1),
    )
    else:
        warp = ObjectWarpingSE3Batch(
            child_part_models[target_part],
            masked_pcls[target_part],
            device,
            canon_labels = canon_labels,
            cost_function = cost_function,
            **cp.deepcopy(PARAM_1),
        )

    child_reconstruction, _, child_params = warp_to_pcd_se3(
        warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
    )

    return child_reconstruction

#define different objective functions
def pca_relational_descriptor(child_part_labels, target_part, shape_weight=10):
    # define shape regularization term
    cost_function = (
        lambda source, target, canon_part_labels, latent_param, scale_param, initial_latents: mask_and_cost_batch_pt(
            target,
            child_part_labels[target_part],
            source,
            canon_part_labels['variational'],
        ) + torch.norm(latent_param - initial_latents) * shape_weight
    )

    return cost_function


#TODO TRY THIS NEXT
def pca_plus_relational_descriptor(variational_child_part_labels, relational_child_part_labels, target_part, shape_weight=10):
    # define shape regularization term
    cost_function = (
        lambda source, target, canon_part_labels, latent_param, scale_param, initial_latents: mask_and_cost_batch_pt(
            target,
            relational_child_part_labels[target_part],
            source,
            canon_part_labels['relational'],
        ) + mask_and_cost_batch_pt(
            target,
            variational_child_part_labels[target_part],
            source,
            canon_part_labels['variational'],
        ) #+ torch.norm(latent_param - initial_latents) * shape_weight
    )

    return cost_function

def baseline_relational_descriptor(child_part_labels, target_part, shape_weight=10):
    # define shape regularization term
    cost_function = (
        lambda source, target, canon_part_labels, latent_param, scale_param, initial_latents: mask_and_cost_batch_pt(
            target,
            child_part_labels[target_part],
            source,
            canon_part_labels['relational'],
        ) + torch.norm(latent_param - initial_latents) * shape_weight
    )

    return cost_function

def no_shape_regularization(child_part_labels, target_part):
    cost_function = (
        lambda source, target, canon_part_labels, latent_param, scale_param, initial_latents: mask_and_cost_batch_pt(
            target,
            child_part_labels[target_part],
            source,
            canon_part_labels['relational'],
        ) 
     )
    return cost_function

#Penalizes warps that change the length x width x height ratio substantially
def scale_preserving_objective(child_part_labels, target_part, shape_weight=10, scale_weight=10):
    cost_function = (
        lambda source, target, canon_part_labels, latent_param, scale_param, initial_latents: mask_and_cost_batch_pt(
            target,
            child_part_labels[target_part],
            source,
            canon_part_labels['relational'],
        ) + torch.norm(latent_param - initial_latents) * shape_weight + torch.std(scale_param, axis=-1) * scale_weight
     )
    return cost_function

def scale_preserving_no_shape_objective(child_part_labels, target_part, scale_weight=10):

    def cost_function(source, target, canon_part_labels, latent_param, scale_param, initial_latents):
        cost = mask_and_cost_batch_pt(
            target,
            child_part_labels[target_part],
            source,
            canon_part_labels,
        ) 

        cost += torch.std(scale_param, axis=-1)* scale_weight
        print(cost)
        return cost

    return cost_function

def pca_double_eval():

    def cost_function(source, target, canon_part_labels, latent_param, scale_param, initial_latents):
        cost = mask_and_cost_batch_pt(
            target,
            child_part_labels[target_part],
            source,
            canon_part_labels,
        ) 
        transformed_pcl = canon_model.pca #do the thing
        new_latents = pca.transform(transformed_pcl)

        return cost

    pass

pcls = []
pca_child_part_labels = get_pca_descriptors(masked_pcls, [target_part])
test_objectives = [ baseline_relational_descriptor(child_part_labels, target_part, shape_weight=1), 
                   pca_relational_descriptor(pca_child_part_labels, target_part, shape_weight=1),
                   pca_plus_relational_descriptor(pca_child_part_labels, child_part_labels, target_part, shape_weight=1),
                   no_shape_regularization(child_part_labels, target_part),
                   scale_preserving_objective(child_part_labels, target_part, shape_weight=10, scale_weight=10),
                   None, 
                   ]
                   
names = ['baseline_relational_descriptor_handle',
         'pca_descriptor_handle',
         'pca_plus_relational_handle',
         'no_shape_regularization',
         'scale_preserving',
         'no_relational_descriptors',


         # 'baseline_relational_descriptor_cup',
         # 'pca_descriptor_cup',
         # 'pca_plus_relational_cup',


         #'scale_preserving',
         #' scale_preserving_no_shape_objective',
         ]

for objective in test_objectives: 
    pcls.append(warp_with_objective(objective))

canon_pcls = {target_part: child_part_models[target_part].canonical_pcl}
canon_labels = {'variational': get_pca_descriptors({target_part: child_part_models[target_part].canonical_pcl}, [target_part]), 'relational': canon_child_part_labels}

scene_pcl_markers = {}
for key in masked_pcls.keys():
    scene_pcl_markers[key] = {
            "size": 5,
            "opacity": 1,
            "colorscale": 'Plotly3'
        }
all_viz_pcls = {}
all_viz_markers = {}
for i in range(len(pcls)):
    name = names[i]
    var_label_target, var_label_target_markers = display_descriptors(masked_pcls, pca_child_part_labels, [target_part], root='var', colors = [('reds', 'blues')])
    var_label_source, var_label_source_markers = display_descriptors({target_part: pcls[i]}, canon_labels['variational'], [target_part], root='canon_var', colors = [('reds', 'blues')])
    rel_label_target, rel_label_target_markers = display_descriptors(masked_pcls, child_part_labels, [target_part], root='rel', colors = [('greens', 'purples')])
    rel_label_source, rel_label_source_markers = display_descriptors({target_part: pcls[i]}, canon_labels['relational'], [target_part], root='canon_rel', colors = [('greens', 'purples')])

    all_viz_markers[name] = var_label_source_markers | var_label_target_markers | rel_label_source_markers | rel_label_target_markers | scene_pcl_markers
    all_viz_pcls[name] = var_label_source | var_label_target | rel_label_source | rel_label_target | masked_pcls


# {names[i] :  |  display_descriptors({target_part: pcls[i]}, canon_labels['variational'], [target_part], root='canon') \
#                         for i in range(len(pcls))},#

rows = 2
cols = 3
viz_utils.show_pcd_grid_plotly(
    rows,
    cols,
    # {names[i] : {target_part: pcls[i], 'target': scene_pcls[target_part] }  for i in range(len(pcls))},#
    # {names[i] : display_descriptors(masked_pcls, child_part_labels, [target_part]) |  display_descriptors({target_part: pcls[i]}, canon_labels['variational'], [target_part], root='canon') \
    #                     for i in range(len(pcls))},#
    all_viz_pcls ,
    names,
    subplot_titles = names,
    markers = all_viz_markers,
    camera_views = [dict(
                            eye=dict(x=1.2, y=-1.2, z=.2),
                            center=dict(x=.8,y=.5,z=0)
                        ) for i in range(rows * cols)]
).show()