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
from src.ndf_interface import get_part_labels, mask_and_cost_batch_pt, get_canon_labels, get_part_labels
from src.object_warping import ObjectWarpingSE3Batch, PARAM_1, warp_to_pcd_se3
from scipy.spatial.transform import Rotation


inference_kwargs =     final_inference_kwargs = {
                            "train_latents": True,
                            "train_scales": True,
                            "train_poses": True,
                        }

# def load_all_shapenet_files(obj_type):
#     cfg = get_eval_cfg_defaults()
#     config_fname = osp.join(
#         path_util.get_rndf_config(), "eval_cfgs", "base_cfg"
#     )  # args.config)
#     if osp.exists(config_fname):
#         cfg.merge_from_file(config_fname)
#     else:
#         log_info(f"Config file {config_fname} does not exist, using defaults")

#     mesh_data_dirs = {
#         "mug": "mug_centered_obj_normalized",
#         # 'bottle': 'bottle_centered_obj_normalized',
#         "bowl": "bowl_centered_obj_normalized",
#         "syn_rack_easy": "syn_racks_easy_obj",
#         # 'syn_container': 'box_containers_unnormalized'
#     }
#     mesh_data_dirs = {
#         k: osp.join(path_util.get_rndf_obj_descriptions(), v)
#         for k, v in mesh_data_dirs.items()
#     }
#     bad_ids = {
#         "syn_rack_easy": [],
#         "bowl": bad_shapenet_bowls_ids_list,
#         "mug": bad_shapenet_mug_ids_list,
#         "bottle": bad_shapenet_bottles_ids_list,
#         "syn_container": [],
#     }

#     upright_orientation_dict = {
#         "mug": common.euler2quat([np.pi / 2, 0, 0]).tolist(),
#         "bottle": common.euler2quat([np.pi / 2, 0, 0]).tolist(),
#         "bowl": common.euler2quat([np.pi / 2, 0, 0]).tolist(),
#         "syn_rack_easy": common.euler2quat([0, 0, 0]).tolist(),
#         "syn_container": common.euler2quat([0, 0, 0]).tolist(),
#     }

#     mesh_names = {}
#     for k, v in mesh_data_dirs.items():
#         # get train samples
#         objects_raw = os.listdir(v)
#         objects_filtered = [
#             fn
#             for fn in objects_raw
#             if (fn.split("/")[-1] not in bad_ids[k] and "_dec" not in fn)
#         ]
#         # objects_filtered = objects_raw
#         total_filtered = len(objects_filtered)
#         train_n = int(total_filtered * 0.9)
#         test_n = total_filtered - train_n

#         # train_objects = sorted(objects_filtered)[:train_n]
#         # test_objects = sorted(objects_filtered)[train_n:]

#         # log_info('\n\n\nTest objects: ')
#         # log_info(test_objects)
#         # # log_info('\n\n\n')

#         mesh_names[k] = objects_filtered
#     print("got em")

#     obj_classes = list(mesh_names.keys())

#     scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
#     scale_default = cfg.MESH_SCALE_DEFAULT

#     # cfg.OBJ_SAMPLE_Y_HIGH_LOW = [0.3, -0.3]
#     cfg.OBJ_SAMPLE_Y_HIGH_LOW = [-0.35, 0.175]
#     x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
#     y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
#     table_z = cfg.TABLE_Z

#     return mesh_names[obj_type]


# def get_mesh(pcl_id):
#     obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
#     mesh = utils.trimesh_load_object(obj_file_path)
#     return mesh


# obj_type = "mug"
# part_labels = {"cup": 37, "handle": 36}
# part_names = ["cup", "handle"]
# directory = (
#     "../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/"
# )


# training_ids = load_all_shapenet_files("mug")
# mug_meshes = {"mug": []}
# for obj_id in training_ids[:1]:
#     mug_meshes["mug"].append(get_mesh(obj_id))

all_meshes = []
root = "~/fewshot/segmentations/"
for i in range(5):
    print(i)
    parts = ["body", "lid", "spout", "handle"]

    mesh_file = {part: f"kettle_{i+1}/kettle_{i+1}_{part}.obj" for part in parts}
    meshes = {part: trimesh.load(root + mesh_file[part]) for part in parts}

    for part in parts:
        utils.trimesh_transform(meshes[part], center=False, scale=1.75)
    all_meshes.append(meshes)
    # mesh_fig = viz_utils.show_meshes_plotly(
    #     {part: meshes[part].vertices for part in parts},
    #     # | {"mug": mug_meshes["mug"][0].vertices},
    #     {part: meshes[part].faces for part in parts},
    #     # | {"mug": mug_meshes["mug"][0].faces},
    # )

meshes = all_meshes[2]
# Sample pcls for part in part
part_pcls = {}
for part in meshes.keys():
    part_pcls[part] = utils.trimesh_create_verts_surface(meshes[part], 1000)


teapot_pcl = viz_utils.show_pcds_plotly(
    {part: part_pcls[part] for part in part_pcls.keys()}
)
teapot_pcl.show()


whole_pcl = utils.trimesh_create_verts_surface(trimesh.util.concatenate(list(meshes.values())), 1000)
whole_pcl = utils.scale_points_circle([whole_pcl], base_scale=.25)[0]

# def get_part_labels(part_pairs, part_names=None):
#     dists = np.sum(
#         np.square(part_pairs[part_names[0]][None] - part_pairs[part_names[1]][:, None]),
#         axis=-1,
#     )

#     part_dists = {part_names[i]: np.min(dists, axis=i) for i in range(len(part_names))}
#     part_labels = {
#         part: np.where(
#             part_dists[part] < np.mean(part_dists[part] * 0.6),
#             np.zeros_like(part_dists[part]),
#             np.ones_like(part_dists[part]),
#         )
#         for part in part_names
#     }

#     return part_labels


# pick a kettle
adjacent_part_pairs = [
    {"body": part_pcls["body"], "lid": part_pcls["lid"]},
    {"body": part_pcls["body"], "spout": part_pcls["spout"]},
    {"body": part_pcls["body"], "handle": part_pcls["handle"]},
    {"handle": part_pcls["handle"], "lid": part_pcls["lid"]},
    {"lid": part_pcls["lid"], "spout": part_pcls["spout"], },

]

ordered_part_pairs = [
    ("body", "lid"),
    ("body", "spout"),
    ("body", "handle"),
    ("handle", "lid"),
    ("lid", "spout")
]

import random, string


def randomword(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


shifted_part_pcls = {}
for key in part_pcls.keys():
    shifted_pcl = part_pcls[key] + .3 * np.mean(part_pcls[key], axis=0)
    shifted_part_pcls[key] = shifted_pcl

all_labeled_parts = {}
for part_pair, ordered_part_pair in zip(adjacent_part_pairs, ordered_part_pairs):
    part_labels = get_part_labels([part_pair])
    # viz_utils.show_pcds_plotly(
    #     {
    #         randomword(5): part_pcls[part][part_labels[part] == 0]
    #         for part in part_pair.keys()
    #     }
    #     | {
    #         randomword(5): part_pcls[part][part_labels[part] == 1]
    #         for part in part_pair.keys()
    #     }
    # ).show()


    all_labeled_parts = (
        all_labeled_parts
        | {
            f"{ordered_part_pair[0]}_{ordered_part_pair[1]}_0_{part}": shifted_part_pcls[part][
                part_labels[part][0] == 0
            ]
            for part in part_pair.keys()
        }
        | {
            f"{ordered_part_pair[0]}_{ordered_part_pair[1]}_1_{part}": shifted_part_pcls[part][
                part_labels[part][0] == 1
            ]
            for part in part_pair.keys()
        }
    )


colors = {
    "body_lid_1_body": "gray",
    "body_lid_0_body": "greens",
    "body_lid_1_lid": "gray",
    "body_lid_0_lid": "greens",
    "body_spout_1_body": "gray",
    "body_spout_0_body": "blues",
    "body_spout_1_spout": "gray",
    "body_spout_0_spout": "blues",
    "body_handle_1_body": "gray",
    "body_handle_0_body": "purples",
    "body_handle_1_handle": "gray",
    "body_handle_0_handle": "purples",
    "handle_lid_1_handle":"gray",
    "handle_lid_0_handle":"turbid",
    "handle_lid_1_lid":"gray",
    "handle_lid_0_lid":"turbid",
    "lid_spout_1_lid":"gray",
    "lid_spout_0_lid":"reds",
    "lid_spout_1_spout":"gray",
    "lid_spout_0_spout":"reds",
}


viz_utils.show_pcds_plotly(all_labeled_parts, colors=colors).show()

teapot_part_model_files = {'body': 'part_based_warp_models/body_dict_20241031-012841_5', 
                          'lid': 'part_based_warp_models/lid_dict_20241031-012841_5',
                          'spout': 'part_based_warp_models/spout_dict_20241031-012841_5',
                          'handle': 'part_based_warp_models/handle_dict_20241031-012841_5', }

teapot_whole_model_file = 'part_based_warp_models/whole_teapot_dict_20241031-012841_5'
teapot_whole_canon_model =  utils.CanonPart.from_pickle(teapot_whole_model_file)


teapot_part_names = ['body', 'handle', 'lid', 'spout']
teapot_part_canon_models = {part: utils.CanonPart.from_pickle(teapot_part_model_files[part]) for part in teapot_part_names}
teapot_reconstructions = {part:None for part in teapot_part_names}

canon_adjacent_part_pairs = [
    {"body": teapot_part_canon_models["body"], "lid": teapot_part_canon_models["lid"]},
    {"body": teapot_part_canon_models["body"], "spout": teapot_part_canon_models["spout"]},
    {"body": teapot_part_canon_models["body"], "handle": teapot_part_canon_models["handle"]},
    {"lid": teapot_part_canon_models["lid"], "spout": teapot_part_canon_models["spout"]}


]
canon_teapot_labels = get_canon_labels(
                canon_adjacent_part_pairs, teapot_part_canon_models, teapot_part_names, 
            )



viz_canon_part_pairs = [{"body": teapot_part_canon_models["body"], "spout": teapot_part_canon_models["spout"]},
{"lid": teapot_part_canon_models["body"], "spout": teapot_part_canon_models["spout"]}]

viz_canon_ordered_part_pairs = [("body", "spout"), ("lid", "spout")]

all_labeled_canon_parts = {}
for part_pair, ordered_part_pair in zip(viz_canon_part_pairs, viz_canon_ordered_part_pairs):
    part_labels = get_canon_labels(
                [part_pair], teapot_part_canon_models, teapot_part_names, 
            )


    all_labeled_canon_parts = (
        all_labeled_canon_parts
        | {
            f"canon_{ordered_part_pair[0]}_{ordered_part_pair[1]}_0_{part}":teapot_part_canon_models[part].canonical_pcl[
                part_labels[part][0] == 0
            ]
            for part in part_pair.keys()
        }
        | {
            f"canon_{ordered_part_pair[0]}_{ordered_part_pair[1]}_1_{part}": teapot_part_canon_models[part].canonical_pcl[
                part_labels[part][0] == 1
            ]
            for part in part_pair.keys()
        }
    )


all_labeled_canon_parts["canon_body_spout_1_spout"], \
all_labeled_canon_parts["canon_body_spout_0_spout"], \
all_labeled_canon_parts["canon_lid_spout_1_spout"], \
all_labeled_canon_parts["canon_lid_spout_0_spout"] = utils.scale_points_circle([all_labeled_canon_parts["canon_body_spout_1_spout"], \
                                                                                all_labeled_canon_parts["canon_body_spout_0_spout"], \
                                                                                all_labeled_canon_parts["canon_lid_spout_1_spout"], \
                                                                                all_labeled_canon_parts["canon_lid_spout_0_spout"]], base_scale=.8)

spout_mean = np.mean(np.concatenate([all_labeled_parts["body_spout_1_spout"],
                                     all_labeled_parts["body_spout_0_spout"],
                                     all_labeled_parts["lid_spout_1_spout"],
                                     all_labeled_parts["lid_spout_0_spout"],]), axis=0)


canon_spout_mean = np.mean(np.concatenate([all_labeled_canon_parts["canon_body_spout_1_spout"], \
                                           all_labeled_canon_parts["canon_body_spout_0_spout"], \
                                           all_labeled_canon_parts["canon_lid_spout_1_spout"], \
                                           all_labeled_canon_parts["canon_lid_spout_0_spout"]]), axis=0)

canon_viz_components = {
    "body_spout_1_spout": all_labeled_parts["body_spout_1_spout"] - spout_mean,
    "body_spout_0_spout": all_labeled_parts["body_spout_0_spout"] - spout_mean,
    "lid_spout_1_spout": all_labeled_parts["lid_spout_1_spout"] - spout_mean,
    "lid_spout_0_spout": all_labeled_parts["lid_spout_0_spout"] - spout_mean,
    } | {
    "canon_body_spout_1_spout": all_labeled_canon_parts["canon_body_spout_1_spout"] - canon_spout_mean,
    "canon_body_spout_0_spout": all_labeled_canon_parts["canon_body_spout_0_spout"] - canon_spout_mean,
    "canon_lid_spout_1_spout": all_labeled_canon_parts["canon_lid_spout_1_spout"] - canon_spout_mean,
    "canon_lid_spout_0_spout": all_labeled_canon_parts["canon_lid_spout_0_spout"] - canon_spout_mean,
    }


# for key in canon_viz_components.keys():
#     canon_viz_components[key] = utils.center_pcl(canon_viz_components[key])


rotation = Rotation.from_euler("zyx", [0., 0., np.pi/8]).as_quat()

for key in ["canon_body_spout_1_spout", "canon_body_spout_0_spout","canon_lid_spout_1_spout", "canon_lid_spout_0_spout"]:
    canon_viz_components[key] = utils.transform_pcd(canon_viz_components[key], utils.pos_quat_to_transform([0,0,0], rotation))



canon_marker = {
            "size": 5,
            "opacity": .05
        }

target_marker = {
            "size": 5,
            "opacity": 1
        }

canon_viz_colors = { \
    "body_spout_1_spout": "gray",
    "body_spout_0_spout": "blues",
    "canon_body_spout_1_spout": "gray",
    "canon_body_spout_0_spout": "blues",
    "lid_spout_1_spout":"gray",
    "lid_spout_0_spout":"reds",
    "canon_lid_spout_1_spout":"gray",
    "canon_lid_spout_0_spout":"reds",
    }

canon_viz_markers = { \
    "body_spout_1_spout": target_marker,
    "body_spout_0_spout": target_marker,
    "canon_body_spout_1_spout": canon_marker ,
    "canon_body_spout_0_spout": canon_marker ,
    "lid_spout_1_spout": target_marker,
    "lid_spout_0_spout": target_marker,
    "canon_lid_spout_1_spout": canon_marker ,
    "canon_lid_spout_0_spout": canon_marker ,
    }



print(all_labeled_canon_parts)


viz_utils.show_pcds_plotly(canon_viz_components, colors=canon_viz_colors, markers=canon_viz_markers).show()

exit(0)

#show the teapot part to be reconstructed with labels, let's do the spout
#show the canon one

teapot_part_labels = get_part_labels(
    adjacent_part_pairs,
)


teapot_part_names = ['spout']



# Reconstructions

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
whole_teapot_reconstruction = None

n_angles = 15


warp = ObjectWarpingSE3Batch(
            teapot_whole_canon_model,
            whole_pcl,
            'cuda',
            **cp.deepcopy(PARAM_1),
        )

teapot_reconstruction, _, _ = warp_to_pcd_se3(
        warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
    )


viz_utils.show_pcds_plotly({'target_teapot': whole_pcl, 'canon_teapot':  teapot_whole_canon_model.canonical_pcl, 'teapot': teapot_reconstruction}).show()

#teapot_part_names = ["body"]
n_angles = 15
for target_part in teapot_part_names:

    canon = teapot_part_canon_models[target_part]

    cost_function = (
        lambda source, target, canon_part_labels: mask_and_cost_batch_pt(
            target,
            teapot_part_labels[target_part],
            source,
            canon_part_labels,
        )
    )
    
    warp = ObjectWarpingSE3Batch(
            canon,
            part_pcls[target_part],
            'cuda',
            canon_labels=canon_teapot_labels[target_part],
            cost_function=cost_function,
            **cp.deepcopy(PARAM_1),
        )
    teapot_reconstructions[target_part], _, _ = warp_to_pcd_se3(
        warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
    )

print("DONE SHOW PLEASE")    


viz_utils.show_pcds_plotly({part: teapot_reconstructions[part] for part in teapot_part_names}).show()

no_rel_teapot_reconstructions = {part:None for part in teapot_part_names}

n_angles = 15
for target_part in teapot_part_names:

    canon = teapot_part_canon_models[target_part]
    warp = ObjectWarpingSE3Batch(
            canon,
            part_pcls[target_part],
            'cuda',
            **cp.deepcopy(PARAM_1),
        )
    no_rel_teapot_reconstructions[target_part], _, _ = warp_to_pcd_se3(
        warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
    )

print("DONE SHOW PLEASE")    


viz_utils.show_pcds_plotly({part: no_rel_teapot_reconstructions[part] for part in teapot_part_names}).show()


viz_utils.show_pcd_grid_plotly(
    1,
    3,
    {"whole": {'target_teapot': whole_pcl, 'canon_teapot':  teapot_whole_canon_model.canonical_pcl, 'teapot': teapot_reconstruction}, 
     "no_rel":{part: teapot_reconstructions[part] for part in teapot_part_names}, 
     "rel": {part: teapot_reconstructions[part] for part in teapot_part_names}},
    ['whole', 'no_rel', 'rel'],
    # colorscales: Optional[List[str]] = None,
    # camera_views: Optional[List[Dict[str, dict]]] = None,
    # save_image: bool = False,
    # save_path: bool = False,
).show()


# #load part models 

# all_labeled_parts = {}
# i = 0
# for part_pair, ordered_part_pair in zip(adjacent_part_pairs, ordered_part_pairs):

#     for part in part_pair.keys():
#         if part == 'body':
#             all_labeled_parts = (
#                 all_labeled_parts
#                 | {
#                     f"{ordered_part_pair[0]}_{ordered_part_pair[1]}_0_{part}": teapot_reconstructions[part][
#                         canon_teapot_labels[part][i] == 0
#                     ]
#                 }
#                 | {
#                     f"{ordered_part_pair[0]}_{ordered_part_pair[1]}_1_{part}": teapot_reconstructions[part][
#                         canon_teapot_labels[part][i] == 1
#                     ]
#                 }
#             )
#             i += 1
#         else:
#             all_labeled_parts = (
#                 all_labeled_parts
#                 | {
#                     f"{ordered_part_pair[0]}_{ordered_part_pair[1]}_0_{part}": teapot_reconstructions[part][
#                         canon_teapot_labels[part][0] == 0
#                     ]
#                 }
#                 | {
#                     f"{ordered_part_pair[0]}_{ordered_part_pair[1]}_1_{part}": teapot_reconstructions[part][
#                         canon_teapot_labels[part][0] == 1
#                     ]
#                 }
#             )

# colors = {
#     "body_lid_0_body": "gray",
#     "body_lid_1_body": "greens",
#     "body_lid_0_lid": "gray",
#     "body_lid_1_lid": "greens",
#     "body_spout_0_body": "gray",
#     "body_spout_1_body": "oranges",
#     "body_spout_0_spout": "gray",
#     "body_spout_1_spout": "oranges",
#     "body_handle_0_body": "gray",
#     "body_handle_1_body": "purples",
#     "body_handle_0_handle": "gray",
#     "body_handle_1_handle": "purples",
# }

# viz_utils.show_pcds_plotly(all_labeled_parts, colors=colors).show()


