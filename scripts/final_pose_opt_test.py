
from src import utils, viz_utils
import numpy as np
import os, os.path as osp
from rndf_robot.utils import util, path_util
import pickle 
import copy as cp
from src.object_warping import ObjectWarpingSE2Batch, ObjectSE2Batch, ObjectSE3Batch, ObjectWarpingSE3Batch, WarpBatch, warp_to_pcd, warp_to_pcd_se2, warp_to_pcd_se3, warp_to_pcd_se3_hemisphere, PARAM_1
import torch
import pytorch3d.ops as ops
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation

import sys 
# get demo info
demo_path  = osp.join(path_util.get_rndf_data(), 'relation_demos', 'release_demos/mug_on_rack_relation')
demo_files = [fn for fn in sorted(os.listdir(demo_path)) if fn.endswith('.npz')]
demos = []
for f in demo_files:
    demo = np.load(demo_path+'/'+f, mmap_mode='r',allow_pickle=True)
    demos.append(demo)

demo_idx = 0
demo = demos[demo_idx]
start_child_pcd = demo['multi_obj_start_pcd'].item()['child']
start_child_pcd = utils.farthest_point_sample(start_child_pcd, 2000)[0]
final_child_pcd = demo['multi_obj_final_pcd'].item()['child']

start_parent_pcd = demo['multi_obj_start_pcd'].item()['parent']
final_parent_pcd = demo['multi_obj_final_pcd'].item()['parent']

pose_constraint_obj = pickle.load(open('temp_combined.pkl', 'rb'))

source_start = demo['multi_obj_start_obj_pose'].item()['child']
source_final = demo['multi_obj_final_obj_pose'].item()['child']

source_start_pos, source_start_quat = source_start[:3], source_start[3:]
source_final_pos, source_final_quat = source_final[:3], source_final[3:]
source_start_trans = utils.pos_quat_to_transform(source_start_pos, source_start_quat)
source_final_trans = utils.pos_quat_to_transform(source_final_pos, source_final_quat)

source_start_to_final = source_final_trans @ np.linalg.inv(source_start_trans)

transformed_start = utils.transform_pcd(start_child_pcd, source_start_to_final)
transformed_final = utils.transform_pcd(final_child_pcd, np.linalg.inv(source_start_to_final))
transformed_constraint = utils.transform_pcd(pose_constraint_obj.canonical_pcl, np.linalg.inv(source_start_to_final))


def cost_batch_pt(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the one-sided Chamfer distance between two batches of point clouds in pytorch."""
    # B x N x K
    diff = torch.sqrt(torch.sum(torch.square(source[:, :, None] - target[:, None, :]), dim=3))
    diff_flat = diff.view(diff.shape[0] * diff.shape[1], diff.shape[2])
    c_flat = diff_flat[list(range(len(diff_flat))), torch.argmin(diff_flat, dim=1)]
    c = c_flat.view(diff.shape[0], diff.shape[1])
    return torch.mean(c, dim=1)


# viz_utils.show_pcds_plotly({'start_child': start_child_pcd , 
#               'start_parent': start_parent_pcd, 
#               'end_child': final_child_pcd , 
#               'end_parent': final_parent_pcd, 
#               'constraint': pose_constraint_obj.canonical_pcl,
#               'transformed_pcd': transformed_start})

#print(cost_batch_pt(torch.from_numpy(np.expand_dims(transformed_start, 0)), torch.from_numpy(np.expand_dims(final_child_pcd, 0))))
# print(cost_batch_pt(torch.from_numpy(np.expand_dims(transformed_constraint, 0)), torch.from_numpy(np.expand_dims(start_child_pcd, 0))))
# print(cost_batch_pt(torch.from_numpy(np.expand_dims(start_child_pcd, 0)), torch.from_numpy(np.expand_dims(transformed_constraint, 0))))

# exit(0)


final_inference_kwargs = {
            "train_latents": False,
            "train_scales": False,
            "train_poses": True
        }

param_1 = cp.deepcopy(PARAM_1)


# pytorch3d.ops.iterative_closest_point()


# def icp(coords, coords_ref, device, n_iter):
#     """
#     Iterative Closest Point
#     """
#     for t in range(n_iter):
#         cdist = torch.cdist(coords - coords.mean(axis=0),
#                             coords_ref - coords_ref.mean(axis=0))
#         mindists, argmins = torch.min(cdist, axis=1)
#         X = torch.linalg.lstsq(coords_ref[argmins], coords)[0]
#         coords = coords.mm(X[:3])
#         rmsd = torch.sqrt((X[3:]**2).sum(axis=1).mean())
#         #print_progress(f'{t+1}/{n_iter}: {rmsd}')
#     return coords

# def print_progress(instr):
#     sys.stdout.write(f'{instr}\r')
#     sys.stdout.flush()

def random_rots(num: int):
    """Sample random rotation matrices."""
    return Rotation.random(num=num).as_quat()

rotations = random_rots(75)

errors = []
transforms = []


random_transform = np.linalg.inv(source_start_to_final)
start_pcl = utils.center_pcl(start_child_pcd)
constr_pcl = utils.center_pcl(utils.transform_pcd(pose_constraint_obj.canonical_pcl, random_transform))

# from difficp import ICP6DoF

# target = torch.from_numpy(start_pcl).unsqueeze(0).float()
# source = torch.from_numpy(constr_pcl).unsqueeze(0).float()
# icp = ICP6DoF(differentiable=True, corr_threshold=10000)
# rigid_pose = icp(source, target)[0].detach().cpu().numpy()
# viz_utils.show_pcds_plotly({'start_child': start_pcl, 
#                      'init': constr_pcl,
#                      'result': utils.transform_pcd(constr_pcl, rigid_pose), })

# exit(0)

# from pycpd import RigidRegistration

# def cpd_transform(source, target, alpha: float=.2):
#     source, target = source.astype(np.float64), target.astype(np.float64)
#     #reg = deformable_registration(**{ 'X': source, 'Y': target, 'tolerance':0.00001 }, alpha=alpha)
#     reg = RigidRegistration(**{ 'X': source, 'Y': target, 'tolerance':0.00001, 'device':'cpu'}, alpha=alpha)
#     reg.register()
#     #Returns the gaussian means and their weights - WG is the warp of source to target
#     q = utils.rotm_to_quat(reg.R)
#     transform = utils.pos_quat_to_transform(reg.t, q)
#     return transform

# source = constr_pcl
# target = start_pcl

# result_transform = cpd_transform(source, target)

# viz_utils.show_pcds_plotly({'start_child': start_pcl, 
#                      'init': constr_pcl,
#                      'result': utils.transform_pcd(constr_pcl, result_transform), })


# import open3d as o3d

# source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(constr_pcl))
# target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(start_pcl))

# reg_p2p = o3d.pipelines.registration.registration_icp(source, target, .01, np.eye(4),
#     o3d.pipelines.registration.TransformationEstimationPointToPoint())
# print(reg_p2p)
# print("Transformation is:")
# print(reg_p2p.transformation)

# viz_utils.show_pcds_plotly({'start_child': start_pcl, 
#                      'init': constr_pcl,
#                      'result': utils.transform_pcd(constr_pcl, reg_p2p.transformation ) , })
# exit(0)

# random_transform = utils.pos_quat_to_transform([0,0,0], rotation)
# random_transform = np.linalg.inv(source_start_to_final)

# viz_utils.show_pcds_plotly({'start_child': start_pcl , 
#                 'result': constr_pcl , })


# result = icp(torch.from_numpy(utils.center_pcl(pose_constraint_obj.canonical_pcl)), torch.from_numpy(utils.center_pcl(start_child_pcd)), 'cpu', 100)
# result = ops.iterative_closest_point(torch.from_numpy(constr_pcl).unsqueeze(0).float(), 
#     torch.from_numpy(start_pcl).unsqueeze(0).float(), max_iterations=500,  relative_rmse_thr=1e-06)
# errors.append(result.rmse)

# # print(result.RTs.R)
# # print(result.RTs.T)
# print(result.converged)
# print(len(result.t_history))
# print(result.t_history)
# icp_quat = utils.rotm_to_quat(result.RTs.R[0])
# icp_transform = utils.pos_quat_to_transform(result.RTs.T, icp_quat)
# transforms.append(icp_transform)

# print(np.linalg.inv(source_start_to_final))

# pcd_result = utils.transform_pcd(constr_pcl, icp_transform)



               
# exit(0)

pose_constraint_obj.canonical_pcl = constr_pcl
combined_warp = ObjectSE3Batch(
                    pose_constraint_obj , [], start_pcl, 'cpu', cost_function=None, **param_1,
                    init_scale=1)
combined_complete, _, combined_params = warp_to_pcd_se3_hemisphere(combined_warp, n_angles=50, n_batches=1, inference_kwargs=final_inference_kwargs)




# fig = go.Figure()

# # Add traces, one for each slider step
# for transform in combined_warp.transform_history:
#     # h_quat = utils.rotm_to_quat(transform.R)
#     # h_t = utils.pos_quat_to_transform(result.RTs.T, h_quat)
#     h_pcl = utils.transform_pcd(pose_constraint_obj.canonical_pcl, transform)
#     fig.add_trace(
#         go.Scatter3d(visible=False, 
#             x=h_pcl[:, 0], y=h_pcl[:, 1], z=h_pcl[:, 2],
#             marker={"size": 5, "color": h_pcl[:, 2], "colorscale": 'viridis'},
#             mode="markers", opacity=1.),)

# fig.add_trace(
#         go.Scatter3d( x=start_pcl[:, 0], y=start_pcl[:, 1], z=start_pcl[:, 2],
#             marker={"size": 5, "color": start_pcl[:, 2], "colorscale": 'plotly3'},
#             mode="markers", opacity=1.),)



# # Make 10th trace visible
# fig.data[10].visible = True

# # Create and add slider
# steps = []
# for i in range(len(fig.data)-1):
#     step = dict(
#         method="update",
#         args=[{"visible": [False] * (len(fig.data)-1) + [True]},
#               {"title": "cost is " + str(combined_warp.cost_history[i])}],  # layout attribute
#     )
#     step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
#     steps.append(step)

# sliders = [dict(
#     active=10,
#     currentvalue={"prefix": "Frequency: "},
#     pad={"t": 50},
#     steps=steps
# )]

# fig.update_layout(
#     sliders=sliders
# )



# fig.show()



#     viz_utils.show_pcds_plotly({'start_child': start_pcl , 
#                   'result': pcd_result , })
#     exit(0)

# icp_transform = transforms[np.argmin(errors)]
# pcd_result = utils.transform_pcd(utils.center_pcl(pose_constraint_obj.canonical_pcl), icp_transform)
# print(errors)



aligned_pcd = pose_constraint_obj.to_transformed_pcd(combined_params)#utils.transform_pcd(constr_pcl, utils.pos_quat_to_transform(combined_params.position, combined_params.quat))#

viz_utils.show_pcds_plotly({'start_child': start_child_pcd, 'result':aligned_pcd})


print(utils.pos_quat_to_transform(combined_params.position, combined_params.quat))

viz_utils.show_pcds_plotly({'start_child': start_child_pcd , 
                'end_child': final_child_pcd , 
                'trans_pcd': utils.transform_pcd(start_child_pcd, np.linalg.inv(utils.pos_quat_to_transform(combined_params.position, combined_params.quat))),
                'aligned_pcd':pose_constraint_obj.to_transformed_pcd(combined_params),
                'end_parent': final_parent_pcd, 
                'constraint': pose_constraint_obj.canonical_pcl})

print(cost_batch_pt(torch.from_numpy(np.expand_dims(aligned_pcd, 0)), torch.from_numpy(np.expand_dims(start_child_pcd, 0))))
print(cost_batch_pt(torch.from_numpy(np.expand_dims(start_child_pcd, 0)), torch.from_numpy(np.expand_dims(aligned_pcd, 0))))




# load pose constraint object 

# load object in start pose
# load target object

#viz to make sure everything is in the right place
#do optimization to regress pose constraint to start pose
#see if it works or what might need to be adjusted to make it work