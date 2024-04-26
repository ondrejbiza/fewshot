
import torch
import open3d as o3d
from src import viz_utils, utils
from src.utils import CanonPart, CanonPartMetadata
from src.object_warping import ObjectWarpingSE3Batch, warp_to_pcd_se3, PARAM_1
import pickle
import copy as cp
import numpy as np

mug_name = 'babyface_mug'
mug_body = f'../ObjectPartSeg/{mug_name}_0_body_only.ply'
mug_handle = f'../ObjectPartSeg/{mug_name}_0_handle_only.ply'
mug_body_and_handle = f'../ObjectPartSeg/{mug_name}_0_handle_and_body.ply'

device = "cuda" if torch.cuda.is_available() else "cpu"
canon_source_scale = 1.0
param_1 = cp.deepcopy(PARAM_1)

handle_o3d = o3d.io.read_point_cloud(mug_handle)

handle_pcl = np.asarray(handle_o3d.points)

#handle_o3d = handle_o3d.select_by_index(np.where(handle_pcl[:,2] < .4))
cl, ind = handle_o3d.remove_statistical_outlier(nb_neighbors=25,std_ratio=2.0)

handle_pcl = handle_pcl[ind]
handle_pcl = handle_pcl[handle_pcl[:,2] < .35]
body_o3d = o3d.io.read_point_cloud(mug_body)

body_pcl = np.asarray(body_o3d.points)
#body_o3d = body_o3d.select_by_index(np.where(body_pcl[:,2] > .4))
cl, ind = body_o3d.remove_statistical_outlier(nb_neighbors=25,std_ratio=2.0)

body_pcl = body_pcl[ind]
body_pcl = body_pcl[body_pcl[:,2] < .35]
# viz_utils.show_pcds_plotly({'before outlier removal': handle_pcl, 'after_outlier_removal': deoutliered_handle_pcl})
# exit(0)

#body_pcl = np.asarray(o3d.io.read_point_cloud(mug_body).points)
#handle_pcl = np.asarray(o3d.io.read_point_cloud(mug_handle).points)
mug_pcl  = np.asarray(o3d.io.read_point_cloud(mug_body_and_handle).points)

handle_model_file = './part_based_warp_models/handle_20240202-160637'
body_model_file = './part_based_warp_models/cup_20240202-160637'

handle_model = pickle.load(open(handle_model_file, 'rb'))
body_model = pickle.load(open(body_model_file, 'rb'))

print(handle_pcl.shape)
print(body_pcl.shape)
downsampled_handle, _ = utils.farthest_point_sample(handle_pcl, 500)
downsampled_cup, _ = utils.farthest_point_sample(body_pcl, 500)
print(downsampled_handle.shape)
print(downsampled_cup.shape)
inference_kwargs = {
            "train_latents": True,
            "train_scales": True,
            "train_poses": True,
        }

handle_warp = ObjectWarpingSE3Batch(
                handle_model,
                downsampled_handle,
                device,
                **param_1,
                init_scale=canon_source_scale,
            )
source_handle_complete, _, source_param = warp_to_pcd_se3(
    handle_warp, n_angles=12, n_batches=15, inference_kwargs=inference_kwargs
)

cup_warp = ObjectWarpingSE3Batch(
                body_model,
                downsampled_cup,
                device,
                **param_1,
                init_scale=canon_source_scale,
            )
source_cup_complete, _, source_param = warp_to_pcd_se3(
    cup_warp, n_angles=12, n_batches=15, inference_kwargs=inference_kwargs
)

viz_utils.show_pcds_plotly({'target': downsampled_handle, 'warp':source_handle_complete,
							'target_cup': downsampled_cup, 'warp_cup':source_cup_complete})