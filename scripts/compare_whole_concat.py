
from src.object_warping import ObjectSE3Batch,ObjectWarpingSE3Batch, ObjectAEWarpingSE3Batch, warp_to_pcd_se3_hemisphere, PARAM_1, mask_and_cost_batch_pt
from src.object_warping_contact import ObjectSE3BatchContact, ObjectWarpingSE3BatchContact, warp_to_pcd_se3_hemisphere_contact
import numpy as np
from src import utils, viz_utils
from rndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from rndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
from rndf_robot.utils import util, path_util
import os, os.path as osp
from airobot.utils import common
from airobot import log_info
import pickle
from scripts.generate_warps import load_all_shapenet_files, get_mesh, get_segmented_mesh, CanonPart, CanonPartMetadata
import copy as cp
from scipy.spatial.transform import Rotation
import torch 
import trimesh

inference_kwargs = {
            "train_latents": True,
            "train_scales": True,
            "train_poses": True,
        }

device = 'cuda'


def optimize_alignment_and_warp(target, source, with_contact=False, target_contacts=[], source_contacts=[], n_angles=8, weight=1):
    if with_contact: 
        cost_function = lambda source, target, canon_points: contact_constraint(source, target, target_contacts, canon_points, weight=weight)
        warp = ObjectWarpingSE3BatchContact(source, source_contacts, target,  device, cost_function=cost_function, **cp.deepcopy(PARAM_1),) 
        result, _, warping_params= warp_to_pcd_se3_hemisphere_contact(warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs) 
    else: 
        warp = ObjectAEWarpingSE3Batch(source, target, device, **cp.deepcopy(PARAM_1),) 
        result, _, warping_params= warp_to_pcd_se3_hemisphere(warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs) 
        print(f"WARPING PARAMS: {warping_params}" )
    return warping_params

def whole_object_warping(target, canonical_object):
    warp_params = optimize_alignment_and_warp(target, canonical_object)
    return canonical_object.to_transformed_pcd(warp_params), canonical_object.to_transformed_mesh(warp_params)



import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_all_pointclouds(pcls, names, target_pcl, target_name, warp_identifier):
    fig = make_subplots(rows=3, cols=4,
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'},],
                               [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'},],
                               [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'},]])

    colorscales = ["Plotly3", "Viridis", "Blues", "Greens", "Greys", "Oranges", "Purples", "Reds"]
    whole_colorscale = 'Greens'
    parts_colorscale = 'Blues'
    target_colorscale = 'Viridis'

    for row in range(1, 4, 1):
        fig.add_trace(
            go.Scatter3d(
                x=target_pcl[:, 0], y=target_pcl[:, 1], z=target_pcl[:, 2],
                marker={"size": 5, "color": target_pcl[:, 0], "colorscale": target_colorscale},
                mode="markers", opacity=1., name=target_name),
            row=row, col=1
        )

        fig.add_trace(
            go.Scatter3d(
                x=target_pcl[:, 0], y=target_pcl[:, 1], z=target_pcl[:, 2],
                marker={"size": 5, "color": target_pcl[:, 0], "colorscale": target_colorscale},
                mode="markers", opacity=1., name=target_name),
            row=row, col=2
        )

        fig.add_trace(
            go.Scatter3d(
                x=pcls[names[0]][:, 0], y=pcls[names[0]][:, 1], z=pcls[names[0]][:, 2],
                marker={"size": 5, "color": pcls[names[0]][:, 0], "colorscale": whole_colorscale},
                mode="markers", opacity=1., name=names[0]),
            row=row, col=2
        )

        fig.add_trace(
            go.Scatter3d(
                x=pcls[names[1]][:, 0], y=pcls[names[1]][:, 1], z=pcls[names[1]][:, 2],
                marker={"size": 5, "color": pcls[names[1]][:, 2], "colorscale": whole_colorscale},
                mode="markers", opacity=1., name=names[1]),
            row=row, col=3
        )


        # fig.add_trace(
        #     go.Scatter3d(
        #         x=pcls[names[1]]['cup'][:, 0], y=pcls[names[1]]['cup'][:, 1], z=pcls[names[1]]['cup'][:, 2],
        #         marker={"size": 5, "color": pcls[names[1]]['cup'][:, 0], "colorscale": parts_colorscale},
        #         mode="markers", opacity=1., name=names[1]),
        #     row=row, col=3
        # )

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=pcls[names[1]]['handle'][:, 0], y=pcls[names[1]]['handle'][:, 1], z=pcls[names[1]]['handle'][:, 2],
        #         marker={"size": 5, "color": pcls[names[1]]['handle'][:, 0], "colorscale": parts_colorscale},
        #         mode="markers", opacity=1., name=names[1]),
        #     row=row, col=3
        # )

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=pcls[names[2]]['cup'][:, 0], y=pcls[names[2]]['cup'][:, 1], z=pcls[names[2]]['cup'][:, 2],
        #         marker={"size": 5, "color": pcls[names[2]]['cup'][:, 2], "colorscale": parts_colorscale},
        #         mode="markers", opacity=1., name=names[2]),
        #     row=row, col=4
        # )

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=pcls[names[2]]['handle'][:, 0], y=pcls[names[2]]['handle'][:, 1], z=pcls[names[2]]['handle'][:, 2],
        #         marker={"size": 5, "color": pcls[names[2]]['handle'][:, 2], "colorscale": parts_colorscale},
        #         mode="markers", opacity=1., name=names[2]),
        #     row=row, col=4
        # )

    # fig.add_trace(
    #     go.Scatter3d(
    #         x=pcls[names[4]]['cup'][:, 0], y=pcls[names[4]]['cup'][:, 1], z=pcls[names[4]]['cup'][:, 2],
    #         marker={"size": 5, "color": pcls[names[4]]['cup'][:, 2], "colorscale": colorscale},
    #         mode="markers", opacity=1., name=names[4]),
    #     row=2, col=3
    # )

    # fig.add_trace(
    #     go.Scatter3d(
    #         x=pcls[names[4]]['handle'][:, 0], y=pcls[names[4]]['handle'][:, 1], z=pcls[names[4]]['handle'][:, 2],
    #         marker={"size": 5, "color": pcls[names[4]]['handle'][:, 2], "colorscale": colorscale},
    #         mode="markers", opacity=1., name=names[4]),
    #     row=2, col=3
    # )


    fw=go.FigureWidget(fig)
    print(fw.layout)

    all_cameras = [fw.layout.scene1.camera, fw.layout.scene2.camera, fw.layout.scene3.camera,
                   fw.layout.scene4.camera, fw.layout.scene5.camera, fw.layout.scene6.camera]


    with fw.batch_update():
        fw.layout.update(width=800, height=600) 
        for camera in [fw.layout.scene1.camera, fw.layout.scene2.camera, fw.layout.scene3.camera, fw.layout.scene4.camera]:
            camera.up=dict(x=0, y=1, z=0)   
            camera.eye=dict(x=2.5, y=1.75, z=1)   

    fw.update_layout(height=600, width=800, title_text=f"Object {target_name} Warping Comparison ")
    #fw.write_image(f"downsampled_contact_warps/demo_target_{target_id}_warp_{warp_identifier}_corner.png")
    fw.show()

    with fw.batch_update():
        fw.layout.update(width=800, height=600) 
        for camera in [fw.layout.scene5.camera, fw.layout.scene6.camera, fw.layout.scene7.camera, fw.layout.scene8.camera,]:
            camera.eye=dict(x=2.5, y=0, z=0)   

    #fw.update_layout(height=600, width=800, title_text=f"Object {target_name} Warping Comparison ")
    #fw.write_image(f"downsampled_contact_warps/demo_target_{target_id}_warp_{warp_identifier}_side.png")
    fw.show()

    with fw.batch_update():
        fw.layout.update(width=800, height=600) 
        for camera in [fw.layout.scene9.camera, fw.layout.scene10.camera, fw.layout.scene11.camera, fw.layout.scene12.camera,]:
            camera.eye=dict(x=0, y=2.5, z=0)   

    fw.update_layout(title_text=f"Object {target_name} Warping Comparison ")
    #fw.write_image(f"downsampled_contact_warps/all_demo_target_{target_id}_warp_{warp_identifier}_top.png")

    # def cam_change(layout, camera):
    #     fw.layout.scene2.camera = camera

    # fw.layout.scene1.on_change(cam_change, 'camera')

    fw.show()


if __name__ == "__main__":

    warp_file_stamp = '20250217-190200_10'

    #todo: generalize for other objects
    object_warp_file = f'./ae_whole_mug_{warp_file_stamp}'
    concat_warp_file = f'./ae_mug_concat_{warp_file_stamp}'

    whole_object_canonical = pickle.load(open( object_warp_file, 'rb'))
    concat_object_canonical = pickle.load(open( concat_warp_file, 'rb'))

    all_shapenet_mugs = load_all_shapenet_files('mug')
    while True: 
        target_id = all_shapenet_mugs[np.random.choice(len(all_shapenet_mugs))]
        if target_id == whole_object_canonical.metadata.canonical_id or target_id in whole_object_canonical.metadata.training_ids:
            continue
        try: 
            #target_id = '387b695db51190d3be276203d0b1a33f'
            target_whole_mesh = get_mesh(target_id)
            #target_part_meshes = get_segmented_mesh(target_id)
            break
        except:
            continue
    print(target_id)
    
    rotation = Rotation.from_euler("yxz", [np.pi * np.random.random(), np.pi * np.random.random(), np.pi * np.random.random()]).as_matrix()

    target_whole = utils.trimesh_create_verts_surface(target_whole_mesh, num_surface_samples=1000)
    whole_warped, _ = whole_object_warping(target_whole, whole_object_canonical)
    concat_warped, _ = whole_object_warping(target_whole, concat_object_canonical)

    names = ['Whole Warped',
             'Concat Warped']

    result_pcls = {'Whole Warped': whole_warped,
                   'Concat Warped': concat_warped}
   
    display_all_pointclouds(result_pcls, names, target_whole, target_id, warp_file_stamp)


