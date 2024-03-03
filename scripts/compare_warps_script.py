
from src.object_warping import ObjectSE3Batch, ObjectWarpingSE3Batch, warp_to_pcd_se3_hemisphere, PARAM_1
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

#For loading the part-segmented shapenet mugs
#That's currently hardcoded into the path
def load_segmented_pointcloud_from_txt(pcl_id, num_points=2048, 
                                       root = 'Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390'):
    fn = os.path.join(root, pcl_id + '.txt')
    cls = 'Mug'
    data = np.loadtxt(fn).astype(np.float32)
    #if not self.normal_channel: #<-- ignore the normals
    point_set = data[:, 0:3]
    #else:
        #point_set = data[:, 0:6]
    seg_ids = data[:, -1].astype(np.int32)
    point_set[:, 0:3] = utils.center_pcl(point_set[:, 0:3])

    #fixed transform to align with the other mugs being used
    rotation = Rotation.from_euler("zyx", [0., np.pi/2, 0.]).as_quat()
    transform = utils.pos_quat_to_transform([0,0,0], rotation)
    point_set = utils.transform_pcd(point_set, transform)

    #choice = np.random.choice(len(seg_ids), num_points, replace=True)
    return point_set, cls, seg_ids


def furthest_point_sample(contacts, pointcloud, num_points):
    sampled_contacts = []
    start_point = contacts[np.random.choice(len(contacts))]
    sampled_contacts.append(start_point)
    distances = np.linalg.norm(pointcloud[contacts] - pointcloud[start_point], axis=0)
    sampled_contacts.append(contacts[np.argmax(distances)])
    return sampled_contacts

def get_contact_points(cup_pcl, handle_pcl, part_names): 
    knns = utils.get_closest_point_pairs_thresh(cup_pcl, handle_pcl, .0004)#get_closest_point_pairs(cup_pcl, handle_pcl, 11)
    viz_utils.show_pcds_plotly({'cup': cup_pcl, 'handle': handle_pcl, })#'pts_1': cup_knns,}) #'pts_2': handle_pcl[knns[:, 0]]})
    return {part_names[0]: knns[:,1 ], part_names[1]: knns[:, 0]}

def cost_batch_pt(source, target): 
    """Calculate the one-sided Chamfer distance between two batches of point clouds in pytorch."""
    # B x N x K
    diff = torch.sqrt(torch.sum(torch.square(source[:, :, None] - target[:, None, :]), dim=3))
    diff_flat = diff.view(diff.shape[0] * diff.shape[1], diff.shape[2])
    c_flat = diff_flat[list(range(len(diff_flat))), torch.argmin(diff_flat, dim=1)]
    c = c_flat.view(diff.shape[0], diff.shape[1])
    return torch.mean(c, dim=1)

def contact_constraint(source, target, source_contact_indices, canon_points, weight=.1):  
    #ax = plt.subplot(111, projection='3d')  
    displayable_target = target.detach().numpy()
    displayable_canon_points = canon_points.detach().numpy()
    #print(displayable_canon_points.shape)
    displayable_source = source.detach().numpy()
    displayable_source_points = displayable_source[:, source_contact_indices, :]
    #print(displayable_source_points.shape)
    constraint = cost_batch_pt(canon_points, source[:, source_contact_indices, :]) * weight
    #print(constraint)
    # for i in range(len(target)):
    #     ax.scatter(displayable_target[i, :, 0], displayable_target[i, :, 1], displayable_target[i, :, 2], alpha=.05)
    #     ax.scatter(displayable_canon_points[i, :, 0], displayable_canon_points[i, :, 1], displayable_canon_points[i, :, 2], color='g')

    # ax.scatter(displayable_source[0, :, 0], displayable_source[0, :, 1], displayable_source[0, :, 2], color='r')
    # ax.scatter(displayable_source_points[0, :, 0], displayable_source_points[0, :, 1], displayable_source_points[0, :, 2], color='y')
    # #ax.scatter(displayable_source_points[0, :, 0], displayable_source_points[0, :, 1], displayable_source_points[0, :, 2], color='r')
    
    # plt.show()
    return cost_batch_pt(source, target) + constraint



#input training set and warp from that training set 

def optimize_alignment_se2(target, source, with_contact=False, target_contacts=[], source_contact=[], n_angles=150, weight=1):
    if with_contact: 
        cost_function = lambda source, target, canon_points: contact_constraint(source, target, target_contacts, canon_points, weight=weight)
        warp = ObjectSE2Batch(source, source_contact, target,  'cpu', cost_function=cost_function, **cp.deepcopy(PARAM_1),) 
        result, _, alignment_params= warp_to_pcd_se2(warp, n_angles, n_batches=1, inference_kwargs=inference_kwargs) 
    else: 
        warp = ObjectSE2Batch(source, [], target,  'cpu', cost_function=None, **cp.deepcopy(PARAM_1),) 
        result, _, alignment_params= warp_to_pcd_se2(warp, n_angles, n_batches=1, inference_kwargs=inference_kwargs) 
    return alignment_params

def optimize_alignment(target, source, with_contact=False, target_contacts=[], source_contact=[], n_angles=150, weight=1):
    if with_contact: 
        cost_function = lambda source, target, canon_points: contact_constraint(source, target, target_contacts, canon_points, weight=weight)
        warp = ObjectSE3Batch(source, source_contact, target,  'cpu', cost_function=cost_function, **cp.deepcopy(PARAM_1),) 
        result, _, alignment_params= warp_to_pcd_se3_hemisphere(warp, n_angles, n_batches=1, inference_kwargs=inference_kwargs) 
    else: 
        warp = ObjectSE3Batch(source, [], target,  'cpu', cost_function=None, **cp.deepcopy(PARAM_1),) 
        result, _, alignment_params= warp_to_pcd_se3_hemisphere(warp, n_angles, n_batches=1, inference_kwargs=inference_kwargs) 
        print(f"ALIGNMENT PARAMS: {alignment_params}" )
    return alignment_params
#
def optimize_alignment_and_warp(target, source, with_contact=False, target_contacts=[], source_contacts=[], n_angles=150, weight=1):
    if with_contact: 
        cost_function = lambda source, target, canon_points: contact_constraint(source, target, target_contacts, canon_points, weight=weight)
        warp = ObjectWarpingSE3Batch(source, source_contacts, target,  'cpu', cost_function=cost_function, **cp.deepcopy(PARAM_1),) 
        result, _, warping_params= warp_to_pcd_se3_hemisphere(warp, n_angles, n_batches=1, inference_kwargs=inference_kwargs) 
    else: 
        warp = ObjectWarpingSE3Batch(source, [], target,  'cpu', cost_function=None, **cp.deepcopy(PARAM_1),) 
        result, _, warping_params= warp_to_pcd_se3_hemisphere(warp, n_angles, n_batches=1, inference_kwargs=inference_kwargs) 
        print(f"WARPING PARAMS: {warping_params}" )
    return warping_params

#nearest_neighbor search (chamfer distance)
#nearest neighbor search (warping)
def whole_object_alignment(target, canonical_object):
    alignment_params = optimize_alignment(target, canonical_object)
    return canonical_object.to_transformed_pcd(alignment_params), canonical_object.to_transformed_mesh(alignment_params)

def whole_object_warping(target, canonical_object):
    warp_params = optimize_alignment_and_warp(target, canonical_object)
    return canonical_object.to_transformed_pcd(warp_params), canonical_object.to_transformed_mesh(warp_params)

#nearest_neighbor search + alignment(chamfer distance)
#nearest neighbor search + alignment (warping)
def part_based_alignment_contact(target_parts, target_contacts, canonical_parts, part_names, downsample_contacts=False, weight=.01):
    aligned_pcls = {}
    aligned_meshes = {}
    for part in part_names:
        target, contacts, canon_obj = target_parts[part], target_contacts[part], canonical_parts[part]
        canon_contacts = canon_obj.contact_points
        if downsample_contacts:
            contacts = furthest_point_sample(target, contacts, 2)
        alignment_params = optimize_alignment(target, canon_obj, True, contacts, canon_contacts, weight=weight)
        aligned_pcls[part] = canon_obj.to_transformed_pcd(alignment_params)
        aligned_meshes[part] = canon_obj.to_transformed_mesh(alignment_params)
    return aligned_pcls, aligned_meshes

def part_based_alignment_no_contact(target_parts, canonical_parts, part_names):
    aligned_pcls = {}
    aligned_meshes = {}
    for part in part_names:
        target, canon_obj = target_parts[part], canonical_parts[part]
        alignment_params = optimize_alignment(target, canon_obj, False)
        aligned_pcls[part] = canon_obj.to_transformed_pcd(alignment_params)
        aligned_meshes[part] = canon_obj.to_transformed_mesh(alignment_params)
    return aligned_pcls, aligned_meshes

def part_based_warping_contact(target_parts, target_contacts, canonical_parts, part_names, downsample_contacts=False, weight=.01):
    warped_pcls = {}
    warped_meshes = {}
    for part in part_names:
        target, contacts, canon_obj = target_parts[part], target_contacts[part], canonical_parts[part]
        canon_contacts = canon_obj.contact_points
        if downsample_contacts:
            contacts = furthest_point_sample(target, contacts, 2)
        alignment_params = optimize_alignment_and_warp(target, canon_obj, True, contacts, canon_contacts, weight=weight)
        warped_pcls[part] = canon_obj.to_transformed_pcd(alignment_params)
        warped_meshes[part] = canon_obj.to_transformed_mesh(alignment_params)
    return warped_pcls, warped_meshes

def part_based_warping_no_contact(target_parts, canonical_parts, part_names):
    aligned_pcls = {}
    aligned_meshes = {}
    for part in part_names:
        target, canon_obj = target_parts[part], canonical_parts[part]
        alignment_params = optimize_alignment_and_warp(target, canon_obj, False)
        aligned_pcls[part] = canon_obj.to_transformed_pcd(alignment_params)
        aligned_meshes[part] = canon_obj.to_transformed_mesh(alignment_params)
    return aligned_pcls, aligned_meshes

#nearest_neighbor search + alignment(chamfer distance)
#nearest neighbor search + alignment (warping)

#part based alignment (with downsampled contact points)
#part based alignment plus warping (with downsampled contact points)
#nearest_neighbor search + alignment(chamfer distance)
#nearest neighbor search + alignment (warping)



import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_all_pointclouds(pcls, names, target_pcl, target_name, warp_identifier):
    fig = make_subplots(rows=3, cols=3,
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'},],
                               [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'},],
                               [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'},]])

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
                x=pcls[names[0]][:, 0], y=pcls[names[0]][:, 1], z=pcls[names[0]][:, 2],
                marker={"size": 5, "color": pcls[names[0]][:, 0], "colorscale": whole_colorscale},
                mode="markers", opacity=1., name=names[0]),
            row=row, col=2
        )

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=pcls[names[1]][:, 0], y=pcls[names[1]][:, 1], z=pcls[names[1]][:, 2],
        #         marker={"size": 5, "color": pcls[names[1]][:, 2], "colorscale": colorscale},
        #         mode="markers", opacity=1., name=names[1]),
        #     row=1, col=3
        # )

        fig.add_trace(
            go.Scatter3d(
                x=pcls[names[1]]['cup'][:, 0], y=pcls[names[1]]['cup'][:, 1], z=pcls[names[1]]['cup'][:, 2],
                marker={"size": 5, "color": pcls[names[1]]['cup'][:, 0], "colorscale": parts_colorscale},
                mode="markers", opacity=1., name=names[1]),
            row=row, col=3
        )

        fig.add_trace(
            go.Scatter3d(
                x=pcls[names[1]]['handle'][:, 0], y=pcls[names[1]]['handle'][:, 1], z=pcls[names[1]]['handle'][:, 2],
                marker={"size": 5, "color": pcls[names[1]]['handle'][:, 0], "colorscale": parts_colorscale},
                mode="markers", opacity=1., name=names[1]),
            row=row, col=3
        )

    # fig.add_trace(
    #     go.Scatter3d(
    #         x=pcls[names[3]]['cup'][:, 0], y=pcls[names[3]]['cup'][:, 1], z=pcls[names[3]]['cup'][:, 2],
    #         marker={"size": 5, "color": pcls[names[3]]['cup'][:, 2], "colorscale": colorscale},
    #         mode="markers", opacity=1., name=names[3]),
    #     row=2, col=2
    # )

    # fig.add_trace(
    #     go.Scatter3d(
    #         x=pcls[names[3]]['handle'][:, 0], y=pcls[names[3]]['handle'][:, 1], z=pcls[names[3]]['handle'][:, 2],
    #         marker={"size": 5, "color": pcls[names[3]]['handle'][:, 2], "colorscale": colorscale},
    #         mode="markers", opacity=1., name=names[3]),
    #     row=2, col=2
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
        for camera in [fw.layout.scene1.camera, fw.layout.scene2.camera, fw.layout.scene3.camera]:
            camera.up=dict(x=0, y=1, z=0)   
            camera.eye=dict(x=2.5, y=1.75, z=1)   

    #fw.update_layout(height=600, width=800, title_text=f"Object {target_name} Warping Comparison ")
    #fw.write_image(f"downsampled_contact_warps/demo_target_{target_id}_warp_{warp_identifier}_corner.png")
    #fw.show()

    with fw.batch_update():
        fw.layout.update(width=800, height=600) 
        for camera in [fw.layout.scene4.camera, fw.layout.scene5.camera, fw.layout.scene6.camera]:
            camera.eye=dict(x=2.5, y=0, z=0)   

    #fw.update_layout(height=600, width=800, title_text=f"Object {target_name} Warping Comparison ")
    #fw.write_image(f"downsampled_contact_warps/demo_target_{target_id}_warp_{warp_identifier}_side.png")
    #fw.show()

    with fw.batch_update():
        fw.layout.update(width=800, height=600) 
        for camera in [fw.layout.scene7.camera, fw.layout.scene8.camera, fw.layout.scene9.camera]:
            camera.eye=dict(x=0, y=2.5, z=0)   

    fw.update_layout(height=1000, width=1000, title_text=f"Object {target_name} Warping Comparison ")
    fw.write_image(f"downsampled_contact_warps/all_demo_target_{target_id}_warp_{warp_identifier}_top.png")

    # def cam_change(layout, camera):
    #     fw.layout.scene2.camera = camera

    # fw.layout.scene1.on_change(cam_change, 'camera')

    fw.show()


if __name__ == "__main__":

    warp_file_stamp = '20240202-160637'

    #todo: generalize for other objects
    object_warp_file = f'whole_mug_{warp_file_stamp}'
    cup_warp_file = f'cup_{warp_file_stamp}'
    handle_warp_file = f'handle_{warp_file_stamp}'

    part_names = ['cup', 'handle']
    part_labels = {'cup': 37, 'handle':36}

    part_canonicals = {}
    whole_object_canonical = pickle.load(open( object_warp_file, 'rb'))
    part_canonicals['cup'] = pickle.load(open( cup_warp_file, 'rb'))
    part_canonicals['handle'] = pickle.load(open( handle_warp_file, 'rb'))
    #this assertion may not be needed in the future but for now it helps us keep stuff straight

    cup_adjustment = utils.pos_quat_to_transform(part_canonicals['cup'].center_transform, (0,0,0,1))
    handle_adjustment = utils.pos_quat_to_transform(part_canonicals['handle'].center_transform, (0,0,0,1))
    adjusted_cup_canon = utils.transform_pcd(part_canonicals['cup'].canonical_pcl, cup_adjustment)
    adjusted_handle_canon = utils.transform_pcd(part_canonicals['handle'].canonical_pcl, handle_adjustment)
    

    ten_cup, ten_handle = utils.scale_points_circle([part_canonicals['cup'].canonical_pcl, part_canonicals['handle'].canonical_pcl], base_scale=10)
    adjusted_cup_canon = utils.transform_pcd(ten_cup, cup_adjustment)
    adjusted_handle_canon = utils.transform_pcd(ten_handle, handle_adjustment)
    contact_cup, contact_handle = utils.scale_points_circle([adjusted_cup_canon, adjusted_handle_canon], base_scale=.1)
    
    canon_contacts = get_contact_points(contact_cup, contact_handle, part_names)

    _, cup_contact_indices = utils.farthest_point_sample(part_canonicals['cup'].canonical_pcl[canon_contacts['cup']], 2)
    _, handle_contact_indices = utils.farthest_point_sample(part_canonicals['handle'].canonical_pcl[canon_contacts['handle']], 2)
    canon_contacts['cup'] = canon_contacts['cup'][cup_contact_indices]
    canon_contacts['handle'] = canon_contacts['handle'][handle_contact_indices]

    part_canonicals['cup'].contact_points = canon_contacts['cup']
    part_canonicals['handle'].contact_points = canon_contacts['handle']

    # print(canon_contacts)
    # exit(0)
    #if a target file and name was passed to the function, use that 
    #you need both

    #if there wasn't, 
    #get all the shapenet pointclouds
    #sample a target from shapenet

    all_shapenet_mugs = load_all_shapenet_files()
    while True: 
        target_id = all_shapenet_mugs[np.random.choice(len(all_shapenet_mugs))]
        if target_id == whole_object_canonical.metadata.canonical_id or target_id in whole_object_canonical.metadata.training_ids:
            continue
        try: 
            target_id='5c7c4cb503a757147dbda56eabff0c47'
            # target_id = whole_object_canonical.metadata.canonical_id
            #target_id = '8b1dca1414ba88cb91986c63a4d7a99a'
            target_whole_mesh = get_mesh(target_id)
            target_part_meshes = get_segmented_mesh(target_id)
            break
        except:
            continue

    #target_id='5c7c4cb503a757147dbda56eabff0c47'
    # #cut the mesh
    # orig_center = cp.deepcopy(target_whole_mesh.centroid)
    # target_whole_mesh = trimesh.intersections.slice_mesh_plane(target_whole_mesh, [1, 0, 0], target_whole_mesh.centroid)
    # #target_whole_mesh.show() 
    # for part in part_names:
    #     target_part_meshes[part_labels[part]] = trimesh.intersections.slice_mesh_plane(target_part_meshes[part_labels[part]], [1, 0, 0], orig_center)
    #     #target_part_meshes[part_labels[part]].show()
    
    rotation = Rotation.from_euler("yxz", [np.pi * np.random.random(), np.pi * np.random.random(), np.pi * np.random.random()]).as_matrix()
    #utils.trimesh_transform(target_whole_mesh, center=False, rotation=rotation)

    # fig = go.Figure(data=[go.Mesh3d(x=target_whole_mesh.vertices[:,0],
    #                                 y=target_whole_mesh.vertices[:,1],
    #                                 z=target_whole_mesh.vertices[:,2], 
    #                                 i=target_whole_mesh.faces[:,0],
    #                                 j=target_whole_mesh.faces[:,1],
    #                                 k=target_whole_mesh.faces[:,2], opacity=0.50)])
    # fig.show()

    # for part in part_names:
    #     utils.trimesh_transform(target_part_meshes[part_labels[part]], center=False, rotation=rotation)
    

        # fig = go.Figure(data=[go.Mesh3d(x=target_part_meshes[part_labels[part]].vertices[:,0],
        #                                 y=target_part_meshes[part_labels[part]].vertices[:,1],
        #                                 z=target_part_meshes[part_labels[part]].vertices[:,2], 
        #                                 i=target_part_meshes[part_labels[part]].faces[:,0],
        #                                 j=target_part_meshes[part_labels[part]].faces[:,1],
        #                                 k=target_part_meshes[part_labels[part]].faces[:,2], opacity=0.50)])
        # fig.show()

    target_whole = utils.trimesh_create_verts_surface(target_whole_mesh, num_surface_samples=2000)
    target_parts = {part:utils.trimesh_create_verts_surface(target_part_meshes[part_labels[part]], num_surface_samples=2000) for part in part_names}
    viz_utils.show_pcds_plotly({"whole": target_whole, "cup": target_parts['cup'], "handle":target_parts['handle']})

    target_contacts = get_contact_points(target_parts['cup'], target_parts['handle'], part_names)
    
    _, cup_contact_indices = utils.farthest_point_sample(target_parts['cup'][target_contacts['cup']], 2)
    _, handle_contact_indices = utils.farthest_point_sample(target_parts['handle'][target_contacts['handle']], 2)
    target_contacts['cup'] = target_contacts['cup'][cup_contact_indices]
    target_contacts['handle'] = target_contacts['handle'][handle_contact_indices]

    # viz_utils.show_pcds_plotly({'cup':target_parts['cup'], 'handle':target_parts['handle'], 
    #                             'cup_contacts':  target_parts['cup'][target_contacts['cup']], 'handle_contacts':  target_parts['handle'][target_contacts['handle']]})
    # exit(0)
    target_parts['cup'], target_parts['handle'] = utils.scale_points_circle([target_parts[part] for part in part_names], base_scale=.1)
    # viz_utils.show_pcds_plotly({'canon_cup': part_canonicals['handle'].canonical_pcl, 'target_handle': target_parts['handle']})

    names = ['Whole Warped',
             #'Part Aligned (no contacts)', 
             'Part Warped', ]
             # 'Part Aligned (contacts)', 
             # 'Part Warped (contacts)']

    #whole alignment only (no contact points)
    whole_warped, _ = whole_object_warping(target_whole, whole_object_canonical)
    #whole_aligned, _ = whole_object_alignment(target_whole, whole_object_canonical)
    #whole warp + alignment (no contact points)
    
    #viz_utils.show_pcds_plotly({'cup': part_canonicals['cup'].canonical_pcl[:len(part_canonicals['cup'].mesh_vertices)]})
    #part based alignment only (no contact points)
    #parts_aligned = part_based_alignment_no_contact(target_parts, part_canonicals, part_names)
    #part based alignment plus warping (no contact points)
    parts_warped, parts_warped_mesh = part_based_warping_no_contact(target_parts, part_canonicals, part_names)

    # fig = go.Figure(data=[go.Mesh3d(x=parts_warped_mesh[part].vertices[:,0],
    #                                     y=parts_warped_mesh[part].vertices[:,1],
    #                                     z=parts_warped_mesh[part].vertices[:,2], 
    #                                     i=parts_warped_mesh[part].faces[:,0],
    #                                     j=parts_warped_mesh[part].faces[:,1],
    #                                     k=parts_warped_mesh[part].faces[:,2], opacity=0.50) for part in part_names])
    

    # fig.show()

    # fig = go.Figure(data=[go.Mesh3d(x=parts_warped_mesh[part].vertices[:,0],
    #                                     y=parts_warped_mesh[part].vertices[:,1],
    #                                     z=parts_warped_mesh[part].vertices[:,2], 
    #                                     i=parts_warped_mesh[part].faces[:,0],
    #                                     j=parts_warped_mesh[part].faces[:,1],
    #                                     k=parts_warped_mesh[part].faces[:,2], opacity=0.50) for part in part_names])
    

    # fig.show()
    # exit(0)

    #part based alignment (with contact points)
    #parts_aligned_contact, parts_aligned_mesh = part_based_alignment_contact(target_parts, target_contacts, part_canonicals, part_names)
    #part based alignment plus warping (with contact points)
    #parts_warped_contact, parts_warped_mesh = part_based_warping_contact(target_parts, target_contacts, part_canonicals, part_names)

    result_pcls = {'Whole Warped': whole_warped,#whole_warped}#whole_aligned, 
                   #'Part Aligned (no contacts)': parts_aligned,
                   'Part Warped': parts_warped,}
                   # 'Part Aligned (contacts)': parts_aligned_contact,
                   # 'Part Warped (contacts)': parts_warped_contact,}
   
    # transform = np.eye(4)
    # transform[:3, :3] = np.linalg.inv(rotation)
    # for name in names:
    #     if type(result_pcls[name]) == type({}):
    #         for part in part_names:
    #             result_pcls[name][part] = utils.transform_pcd(result_pcls[name][part], transform)
    #     else:
    #         result_pcls[name] = utils.transform_pcd(result_pcls[name], transform)

    display_all_pointclouds(result_pcls, names, target_whole, target_id, warp_file_stamp)

