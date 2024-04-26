import trimesh
import os
from src import viz_utils, utils
import numpy as np
from src.object_warping import ObjectWarpingSE2Batch, warp_to_pcd_se2, PARAM_1
import pickle
import torch
import copy as cp

inference_kwargs = {
            "train_latents": True,  # not self.ablate_no_warp,
            "train_scales": True,  # not self.ablate_no_scale,
            "train_poses": True,  # not self.ablate_no_pose_training
        }

def show_all_syn_mug_tree_segmentations():
    syn_rack_path = "../relational_ndf/src/rndf_robot/descriptions/objects/syn_racks_easy_obj/"#{pcl_id}/models/model_normalized.obj"
    objects_raw = os.listdir(syn_rack_path) 
    objects_filtered = [fn for fn in objects_raw if '_dec' not in fn]

    for obj in objects_filtered:
        rack_mesh = trimesh.load(syn_rack_path + obj)
        pcl = rack_mesh.vertices

        #Get the top points 
        max_z = np.max(pcl[:,2]) - .01
        max_pts = pcl[pcl[:,2] > max_z]
        center = np.mean(max_pts, 0)

        rad = max(np.linalg.norm(max_pts-center, axis=-1))

        rack_pts = utils.trimesh_create_verts_surface(rack_mesh, 1000)
        trunk = rack_pts[np.linalg.norm(rack_pts[:, :2] - center[:2], axis=-1) < rad]
        branch = rack_pts[np.linalg.norm(rack_pts[:, :2]- center[:2], axis=-1) > rad]

        viz_utils.show_pcds_plotly({'trunk': trunk, 'branch': branch, 'max':max_pts})
        input("cont?")

def real_mug_tree_seg(filename):
    #load file
    pcl = np.load(filename)
    viz_utils.show_pcds_plotly({"real_tree":pcl})

    max_z = np.max(pcl[:,2]) - .015
    max_pts = pcl[pcl[:,2] > max_z]
    center = np.mean(max_pts, 0)

    max_r = np.argmax(np.linalg.norm(pcl[:, :2]-center[:2], axis=-1))
    max_r_multiple = np.argpartition(np.linalg.norm(pcl[:, :2], axis=-1), -2)[-2:]

    normal = pcl[max_r,:2]/np.linalg.norm(pcl[max_r,:2])
    normal_3d = np.array([normal[0], normal[1], 0])

    normal_angle = np.arctan2(normal[1], normal[0])

    rad = np.max(np.linalg.norm(max_pts[:,:2]-center[:2], axis=-1))*5/8
    plane_point = [rad*np.cos(normal_angle)+center[0],rad*np.sin(normal_angle)+center[1],0]
    signs = np.dot(pcl-plane_point, normal_3d)

    


    # x = np.linspace(-.5, .5, 100)

    # y = normal[0]/(-normal[1])*(x-(rad*np.cos(normal_angle)+center[0])) + (rad*np.sin(normal_angle)+center[1])

    # import plotly.graph_objects as go

    # fig = go.Figure(
    # data=[  go.Scatter3d(
    #         x=pcl[:, 0],
    #         y=pcl[:, 1],
    #         z=pcl[:, 2],
    #         mode='markers',
    #         # color = pcl[:, 2],
    #         # colorscale="Viridis",
    #         ), go.Scatter3d(x=x, y=y, z=[.5]*100),
    #         go.Scatter3d(x=pcl[max_r_multiple][:,0], y=pcl[max_r_multiple][:,1], z=pcl[max_r_multiple][:,2]), 
    #         go.Scatter3d(x=[pcl[max_r][0]], y=[pcl[max_r][1]], z=[pcl[max_r][2]]),
    #         go.Scatter3d(x=max_pts[:, 0], y=max_pts[:, 1], z=max_pts[:, 2], )]
    #     )
    # fig.show()
    # exit(0)


    rack_pts = pcl#utils.trimesh_create_verts_surface(rack_mesh, 1000)
    trunk = rack_pts[signs < 0]#rack_pts[np.linalg.norm(rack_pts[:, :2] - center[:2], axis=-1) < rad]
    branch = rack_pts[signs > 0]#rack_pts[np.linalg.norm(rack_pts[:, :2]- center[:2], axis=-1) > rad]
    

    viz_utils.show_pcds_plotly({'trunk': trunk, 'branch': branch, 'max':max_pts, 'max_r':np.atleast_2d(pcl[max_r,:])})
    return {'trunk': trunk, 'branch':branch}, pcl



if __name__ == "__main__":
    #load the file and display
    real_pcl_file = "./real_pcl_data/point_depth_20240423-003009.npy"
    real_pcl = np.load(real_pcl_file)
    viz_utils.show_pcds_plotly({'real': real_pcl.reshape((-1, 3))})


    # target_pcd_parts, pcl = real_mug_tree_seg("./data/test_data/real_tree_real_pc.npy")

    # warp_file_stamp = "20240412-042732"
    # trunk_warp_file = f"part_based_warp_models/trunk_{warp_file_stamp}"
    # branch_warp_file = f"part_based_warp_models/branch_{warp_file_stamp}"
    # target_part_canonicals = {}
    # target_part_canonicals["trunk"] = pickle.load(open(trunk_warp_file, "rb"))
    # target_part_canonicals["branch"] = pickle.load(open(branch_warp_file, "rb"))
    # target_part_names = ['branch', 'trunk']

    # neighbor_data = pickle.load(open('neighbor_state_dict.pkl', 'rb'))

    # target_parts = {}
    # target_params = {}
    # for part in target_part_names:
    #     n_angles = 8

    #     warp = ObjectWarpingSE2Batch(
    #         target_part_canonicals[part],
    #         target_pcd_parts[part],
    #         "cuda" if torch.cuda.is_available() else "cpu",
    #         **cp.deepcopy(PARAM_1),
    #     )
    #     target_parts[part], _, target_params[part] = warp_to_pcd_se2(
    #         warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
    #     )

    # viz_utils.show_pcds_plotly({'trunk': target_parts['trunk'], 
    #                             'branch': target_parts['branch'], 
    #                             'target': pcl,
    #                             'contact_pts_cup':target_parts['branch'][neighbor_data['target_indices']['cup']['branch']], 
    #                             'contact_pts_cup_trunk':target_parts['trunk'][neighbor_data['target_indices']['cup']['trunk']]})





