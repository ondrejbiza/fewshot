import os, os.path as osp
from src import utils, viz_utils
from rndf_robot.utils import util, path_util
from rndf_robot.share.globals import (
    bad_shapenet_mug_ids_list,
    bad_shapenet_bowls_ids_list,
    bad_shapenet_bottles_ids_list,
)
from rndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from airobot import log_info
from airobot.utils import common
import numpy as np
from PIL import Image, ImageDraw
from sklearn.manifold import TSNE
import copy as cp
from sklearn import neighbors
import plotly.express as px
import pickle
from src.object_warping import PARAM_1, ObjectWarpingSE3Batch, warp_to_pcd_se3
from PIL import Image
import webbrowser

import plotly
from plotly.io._base_renderers import BaseHTTPRequestHandler, HTTPServer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


canon_source_scale = 1
mesh_data_dirs = {
    "mug": "mug_centered_obj_normalized",
    # 'bottle': 'bottle_centered_obj_normalized',
    'bowl': 'bowl_centered_obj_normalized',
    'syn_rack_easy': 'syn_racks_easy_obj',
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


from scipy.spatial.transform import Rotation


def load_segmented_pointcloud_from_txt(
    pcl_id,
    num_points=2048,
    root="Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390",
):
    fn = os.path.join(root, pcl_id + ".txt")
    cls = "Mug"
    data = np.loadtxt(fn).astype(np.float32)
    point_set = data[:, 0:3]  # <-- ignore the normals
    seg_ids = data[:, -1].astype(np.int32)
    point_set[:, 0:3] = utils.center_pcl(point_set[:, 0:3])

    # fixed transform to align with the other mugs being used
    rotation = Rotation.from_euler("zyx", [0.0, np.pi / 2, 0.0]).as_quat()
    transform = utils.pos_quat_to_transform([0, 0, 0], rotation)
    point_set = utils.transform_pcd(point_set, transform)
    # point_set = utils.scale_points_circle([point_set], base_scale=0.1)[0]

    return point_set, cls, seg_ids


def check_segmentation_exists(pcl_id):
    root = "Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390"
    fn = os.path.join(root, pcl_id + ".txt")
    return os.path.exists(fn)


def get_syn_rack_images(mesh_files):
    for pcl_id in mesh_files:
        obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/syn_racks_easy_obj/{pcl_id}.obj"
        mesh = utils.trimesh_load_object(obj_file_path)
        # load mesh
        fig = viz_utils.show_meshes_plotly(
            {"mesh": mesh.vertices},
            {"mesh": mesh.faces},
            center=True,
            axis_visible=False,
            background_visible=False,
            camera={"up": dict(x=0, y=1, z=0), "eye": dict(x=2.5, y=1.75, z=1)},
            show_legend=False,
            show=False,
        )
        fig.write_image(f"mesh_images/syn_racks/{pcl_id}.png")
        # input("continue?")
        # save image somewhere

def get_bowl_images(mesh_files):
    for pcl_id in mesh_files:
        obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/bowl_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
        mesh = utils.trimesh_load_object(obj_file_path)
        # load mesh
        fig = viz_utils.show_meshes_plotly(
            {"mesh": mesh.vertices},
            {"mesh": mesh.faces},
            center=True,
            axis_visible=False,
            background_visible=False,
            camera={"up": dict(x=0, y=1, z=0), "eye": dict(x=2.5, y=1.75, z=1)},
            show_legend=False,
            show=False,
        )
        fig.write_image(f"mesh_images/bowls/{pcl_id}.png")
        # input("continue?")
        # save image somewhere

def get_mug_images(mesh_files):
    for pcl_id in mesh_files:
        obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
        mesh = utils.trimesh_load_object(obj_file_path)
        # load mesh
        fig = viz_utils.show_meshes_plotly(
            {"mesh": mesh.vertices},
            {"mesh": mesh.faces},
            center=True,
            axis_visible=False,
            background_visible=False,
            camera={"up": dict(x=0, y=1, z=0), "eye": dict(x=2.5, y=1.75, z=1)},
            show_legend=False,
            show=False,
        )
        fig.write_image(f"mesh_images/mugs/{pcl_id}.png")
        # input("continue?")
        # save image somewhere


def get_segmented_mesh(pcl_id):
    seg_pcl, _, seg_ids = load_segmented_pointcloud_from_txt(pcl_id)
    obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
    unseg_mesh = utils.trimesh_load_object(obj_file_path)
    utils.trimesh_transform(unseg_mesh, center=True, scale=None, rotation=None)
    unseg_pcl, _ = utils.trimesh_get_vertices_and_faces(unseg_mesh)

    X = seg_pcl
    Y = seg_ids
    clf = neighbors.KNeighborsClassifier(2)  # svm.SVC()
    clf.fit(X, Y)

    part_meshes = {}
    for part_label in clf.classes_:
        part_mesh = cp.deepcopy(unseg_mesh)
        unseg_ids = clf.predict(unseg_pcl)
        mask = unseg_ids == part_label
        face_mask = mask[part_mesh.faces].all(axis=1)
        part_mesh.update_faces(face_mask)
        part_mesh.remove_unreferenced_vertices()
        part_meshes[part_label] = part_mesh
    return part_meshes


def get_all_object_warps(mesh_files):
    meshes = []
    for pcl_id in mesh_files:
        obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
        mesh = utils.trimesh_load_object(obj_file_path)
        meshes.append(mesh)

    num_surface_samples = 5000
    small_surface_points = []
    surface_points = []
    mesh_points = []
    hybrid_points = []
    faces = []
    centers = []

    for mesh in meshes:
        translation_matrix = np.eye(4)
        t = mesh.centroid
        sp = utils.trimesh_create_verts_surface(
            mesh, num_surface_samples=num_surface_samples
        )
        ssp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
        mp, f = utils.trimesh_get_vertices_and_faces(mesh)
        ssp, sp, mp, t = utils.scale_points_circle(
            [ssp, sp, mp, np.atleast_2d(t)], base_scale=0.1
        )
        h = np.concatenate([mp, sp])  # Order important!
        translation_matrix[:3, 3] = t.squeeze()
        centers.append(t)
        small_surface_points.append(ssp)
        surface_points.append(sp)
        mesh_points.append(mp)
        faces.append(f)
        hybrid_points.append(h)

    canonical_idx = utils.sst_pick_canonical(hybrid_points)
    print(f"Canonical obj index: {canonical_idx}.")
    # print(mesh_files[canonical_idx])
    # exit(0)
    # canonical_idx = 93

    tmp_obj_points = cp.copy(small_surface_points)
    tmp_obj_points[canonical_idx] = hybrid_points[canonical_idx]

    warps, _ = utils.warp_gen(canonical_idx, tmp_obj_points, alpha=0.01)
    warps = np.atleast_2d(warps)
    warps = np.insert(warps, canonical_idx, np.zeros_like(warps[0]), axis=0)

    np.save("all_mug_warps.pkl", warps)


def get_all_part_warps(mesh_files, part_names):
    all_part_meshes = {part: [] for part in part_names}
    part_labels = {"cup": 37, "handle": 36}
    for pcl_id in mesh_files:
        # obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
        # mesh = utils.trimesh_load_object(obj_file_path)
        # whole_meshes.append(mesh)
        part_meshes = get_segmented_mesh(pcl_id)
        for part in part_names:
            all_part_meshes[part].append(part_meshes[part_labels[part]])

    for part in part_names:
        num_surface_samples = 5000
        small_surface_points = []
        surface_points = []
        mesh_points = []
        hybrid_points = []
        faces = []
        centers = []
        meshes = all_part_meshes[part]

        for mesh in meshes:
            translation_matrix = np.eye(4)
            t = mesh.centroid
            sp = utils.trimesh_create_verts_surface(
                mesh, num_surface_samples=num_surface_samples
            )
            ssp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
            mp, f = utils.trimesh_get_vertices_and_faces(mesh)
            ssp, sp, mp, t = utils.scale_points_circle(
                [ssp, sp, mp, np.atleast_2d(t)], base_scale=0.1
            )
            h = np.concatenate([mp, sp])  # Order important!
            translation_matrix[:3, 3] = t.squeeze()
            centers.append(t)
            small_surface_points.append(ssp)
            surface_points.append(sp)
            mesh_points.append(mp)
            faces.append(f)
            hybrid_points.append(h)

        canonical_idx = utils.sst_pick_canonical(hybrid_points)
        print(f"Canonical obj index: {canonical_idx}.")
        # print(mesh_files[canonical_idx])
        # exit(0)
        # canonical_idx = 93

        tmp_obj_points = cp.copy(small_surface_points)
        tmp_obj_points[canonical_idx] = hybrid_points[canonical_idx]

        warps, _ = utils.warp_gen(canonical_idx, tmp_obj_points, alpha=0.01)
        warps = np.atleast_2d(warps)
        warps = np.insert(warps, canonical_idx, np.zeros_like(warps[0]), axis=0)

        np.save(f"all_{part}_warps.pkl", warps)


def get_whole_reconstructions(canon_obj, part_names, file_string, mesh_files):
    inference_kwargs = {
        "train_latents": True,
        "train_scales": True,
        "train_poses": True,
    }
    param_1 = cp.deepcopy(PARAM_1)
    device = "cuda"
    whole_meshes = []
    # meshes = []
    # for pcl_id in mesh_files:
    #     obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
    #     mesh = utils.trimesh_load_object(obj_file_path)
    #     meshes.append(mesh)

    all_part_meshes = {part: [] for part in part_names}
    part_labels = {"cup": 36, "handle": 37}
    for pcl_id in mesh_files:
        obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
        mesh = utils.trimesh_load_object(obj_file_path)
        whole_meshes.append(mesh)
        part_meshes = get_segmented_mesh(pcl_id)
        for part in part_names:
            all_part_meshes[part].append(part_meshes[part_labels[part]])

    num_surface_samples = 1000
    target_points = []

    for cup_mesh, handle_mesh in zip(all_part_meshes['cup'], all_part_meshes['handle']):
        sp = np.concatenate((utils.trimesh_create_verts_surface(
            cup_mesh, num_surface_samples=num_surface_samples
        ),
        utils.trimesh_create_verts_surface(
            handle_mesh, num_surface_samples=num_surface_samples
        ),))

        target_points.append(sp)

    params = []
    costs = []

    for i in range(len(target_points)):
        target = target_points[i]
        warp = ObjectWarpingSE3Batch(
            canon_obj,
            target,
            device,
            **param_1,
            init_scale=canon_source_scale,
        )
        warped_complete, warp_costs, warp_param = warp_to_pcd_se3(
            warp, n_angles=12, n_batches=15, inference_kwargs=inference_kwargs
        )
        params.append(warp_param)
        costs.append(min(warp_costs))

        warped_mesh = canon_obj.to_transformed_mesh(warp_param)

        fig = viz_utils.show_meshes_plotly(
        {"mesh": warped_mesh.vertices, 'orig': whole_meshes[i].vertices},
        {"mesh": warped_mesh.faces, 'orig': whole_meshes[i].faces},
        center=True,
        axis_visible=False,
        background_visible=False,
        camera={"up": dict(x=0, y=1, z=0), "eye": dict(x=2.5, y=1.75, z=1)},
        show_legend=True,
        show=True,
        )
        fig.write_image(f"mesh_images/mugs/whole_{pcl_id}.png")


    pickle.dump(params, open(f"whole_mug_params_resampled_{file_string}", "wb"))
    np.save(f"whole_mug_costs_resampled_{file_string}", np.array(costs))
    return params, costs


def get_part_reconstructions(canon_parts, part_names, file_string, mesh_files):
    inference_kwargs = {
        "train_latents": True,
        "train_scales": True,
        "train_poses": True,
    }
    param_1 = cp.deepcopy(PARAM_1)
    device = "cuda"
    whole_meshes = []

    num_surface_samples = 1000
    all_part_meshes = {part: [] for part in part_names}
    all_part_targets = {part: [] for part in part_names}
    part_labels = {"cup": 37, "handle": 36}
    for pcl_id in mesh_files:
        obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
        mesh = utils.trimesh_load_object(obj_file_path)
        whole_meshes.append(mesh)
        part_meshes = get_segmented_mesh(pcl_id)
        for part in part_names:
            all_part_meshes[part].append(part_meshes[part_labels[part]])
            all_part_targets[part].append(utils.trimesh_create_verts_surface(
                all_part_meshes[part][-1], num_surface_samples=num_surface_samples
            ))

    part_params = {part: [] for part in part_names}
    part_costs = {part: [] for part in part_names}


    # viz_utils.show_pcds_plotly({'canonical':canon_parts['cup'].canonical_pcl, 'target': all_part_targets['cup'][0]})
    # exit(0)
    for cup_target, handle_target, whole_mesh in zip(all_part_targets['cup'], all_part_targets['handle'], whole_meshes):
        meshes = all_part_meshes[part]

        cup_warp = ObjectWarpingSE3Batch(
            canon_parts['cup'],
            cup_target,
            device,
            **param_1,
            init_scale=1.75,
        )
        warped_complete, cup_costs, cup_param = warp_to_pcd_se3(
            cup_warp, n_angles=12, n_batches=15, inference_kwargs=inference_kwargs
        )

        handle_warp = ObjectWarpingSE3Batch(
            canon_parts['handle'],
            handle_target,
            device,
            **param_1,
            init_scale=1.75,
        )
        warped_complete, handle_costs, handle_param = warp_to_pcd_se3(
            handle_warp, n_angles=12, n_batches=15, inference_kwargs=inference_kwargs
        )


        part_params['cup'].append(cup_param)
        part_costs['cup'].append(min(cup_costs))
        part_params['handle'].append(handle_param)
        part_costs['handle'].append(min(handle_costs))

        warped_cup = canon_parts['cup'].to_transformed_pcd(cup_param)
        warped_handle = canon_parts['handle'].to_transformed_pcd(handle_param)

        # fig = viz_utils.show_pcds_plotly( ) 

        #viz_utils.show_pcds_plotly({'cup': warped_cup, 'handle': warped_handle, 'cup_target': cup_target, 'handle_target': handle_target})

        # fig = viz_utils.show_meshes_plotly(
        #     {"cup": warped_cup.vertices, 'handle': warped_handle.vertices, 'orig': whole_mesh.vertices},
        #     {"cup": warped_cup.faces, 'handle': warped_handle.faces, 'orig': whole_mesh.faces},
        #     center=True,
        #     axis_visible=False,
        #     background_visible=False,
        #     camera={"up": dict(x=0, y=1, z=0), "eye": dict(x=2.5, y=1.75, z=1)},
        #     show_legend=True,
        #     show=True,
        # )
        # fig.write_image(f"mesh_images/mugs/whole_{pcl_id}.png")

    pickle.dump(part_params, open(f"{part}_params_{file_string}", "wb"))
    np.save(f"cup_costs_{file_string}", np.array(part_costs['cup']))
    np.save(f"handle_costs_{file_string}", np.array(part_costs['handle']))

    return part_params, part_costs


def circlify_image(img):
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    # npImage=np.array(img)
    # h,w=img.size

    # # Create same size alpha layer with circle
    # alpha = Image.new('L', img.size,0)
    # draw = ImageDraw.Draw(alpha)
    # draw.pieslice([h*1/4,w*1/4,h*3/4,w*3/4],0,360,fill=255)

    # # Convert alpha Image to numpy array
    # npAlpha=np.array(alpha)
    # # Add alpha layer to RGB
    # npImage=np.dstack((npImage[:,:,:3],npAlpha))

    # # Save with alpha
    return img  # Image.fromarray(npImage)


def color_adjust_image(img, cost, cost_thresh, idx, training_ids):
    img = np.array(img)
    if idx in training_ids:
        img[:, :, 0] = img[:, :, 0]//2
        img[:, :, 1] = img[:, :, 1]//2
        img[:, :, 2] = np.ones_like(img[:, :, 3]) * 200
        img[:, :, 3] = np.where(img[:,:,3] != 0, np.ones_like(img[:, :, 3]) * 50,  img[:, :, 3])
        return Image.fromarray(img)

    if cost is not None and cost >= cost_thresh:
        img[:, :, 0] = img[:, :, 0]//2
        img[:, :, 2] = img[:, :, 1]//2
        img[:, :, 1] = np.ones_like(img[:, :, 3]) * 200
        # img += np.array([0, 100, 0, 0 ], dtype='uint8')
        # img = np.clip(img, 0, 255)
    else:
        img[:, :, 1] = img[:, :, 0]//2
        img[:, :, 2] = img[:, :, 1]//2
        img[:, :, 0] = np.ones_like(img[:, :, 3]) * 200
        #img[:, :, 2] = np.zeros_like(img[:, :, 0])
        # img += np.array([100, 0, 0, 0], dtype='uint8')
        # img = np.clip(img, 0, 255)
    return Image.fromarray(img)


# visualize all the things, show pcl id on hover?
# get the pcl ids closest to
def get_part_embeddings(part_embeddings, part_names, pcl_ids):
    # embedding_info = None
    # for part in part_names:
    #     X_embedded = TSNE(
    #         n_components=1, learning_rate="auto", init="random", perplexity=3
    #     ).fit_transform(part_warps[part])
    #     if embedding_info is None:
    #         embedding_info = X_embedded
    #     else:
    #         embedding_info = np.concatenate([embedding_info, X_embedded], axis=1)
    
    embedding_info = np.concatenate([np.atleast_2d(part_embeddings['cup'][:, 0]).T, np.atleast_2d(part_embeddings['handle'][:,0]).T], axis=1)
    reg = LinearRegression()
    reg.fit(embedding_info[:,0].reshape(-1,1), embedding_info[:,1].reshape(-1,1))

    residuals = embedding_info[:, 1]/reg.coef_ - embedding_info[:, 0] - reg.intercept_/reg.coef_

    diagonal_idxs = np.argpartition(np.abs(residuals[0]), 25)[:25]
    print(diagonal_idxs)
    fig = px.scatter(
        x=embedding_info[:, 0],
        y=embedding_info[:, 1],
        hover_name=[i for i in range(len(pcl_ids))],
    )
    fig.update_traces(marker_color="rgba(0,0,0,0)")

    x_range = np.max(embedding_info[:, 0]) - np.min(embedding_info[:, 0])
    y_range = np.max(embedding_info[:, 0]) - np.min(embedding_info[:, 0])

    for i in range(len(pcl_ids)):
        pcl_id, embedding = pcl_ids[i], embedding_info[i]
        fig.add_layout_image(
            dict(
                source=color_adjust_image(
                    circlify_image(Image.open(f"mesh_images/mugs/{pcl_id}.png")),
                    None,
                    None,
                    i, [43,98,103,91,8,88,49,100,96,97],
                ),
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                x=embedding[0],
                y=embedding[1],
                sizex=x_range / 4,
                sizey=y_range /4,
                sizing="contain",
                opacity=0.8,
                layer="above",
            )
        )
    fig.update_annotations(font_size=25)
    fig.update_layout(xaxis=dict(
        title=dict(
            text="v1"
        )
    ),)

    fig.update_layout(yaxis=dict(
        title=dict(
            text="v2"
        )
    ),)

    fig.show()


def get_object_embeddings(object_embeddings, pcl_ids):
    # embedding_info = TSNE(
    #     n_components=2, learning_rate="auto", init="random", perplexity=3
    # ).fit_transform(object_warps)

    embedding_info = object_embeddings
    fig = px.scatter(
        x=embedding_info[:, 0],
        y=embedding_info[:, 1],
    )

    fig.update_traces(marker_color="rgba(0,0,0,0)")

    x_range = np.max(embedding_info[:, 0]) - np.min(embedding_info[:, 0])
    y_range = np.max(embedding_info[:, 0]) - np.min(embedding_info[:, 0])

    for pcl_id, embedding in zip(pcl_ids, embedding_info):
        fig.add_layout_image(
            dict(
                source=circlify_image(Image.open(f"mesh_images/mugs/{pcl_id}.png")),
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                x=embedding[0],
                y=embedding[1],
                sizex=x_range / 4,
                sizey=y_range / 4,
                sizing="contain",
                opacity=0.8,
                layer="above",
            )
        )
    fig.show()


def get_two_sided_chamfer_distance():
    pass  

def get_part_cost_graph(part_embeddings, part_costs, cost_thresh, pcl_ids, training_ids):
    # embedding_info = None
    # for part in part_names:
    #     X_embedded = TSNE(
    #         n_components=1, learning_rate="auto", init="random", perplexity=3
    #     ).fit_transform(part_warps[part])
    #     if embedding_info is None:
    #         embedding_info = X_embedded
    #     else:
    #         embedding_info = np.concatenate([embedding_info, X_embedded], axis=1)
    print(training_ids)
    embedding_info = np.concatenate([np.atleast_2d(part_embeddings['cup'][:, 0]).T, np.atleast_2d(part_embeddings['handle'][:,0]).T], axis=1)
    

    fig = px.scatter(
        x=embedding_info[:,0], #part_embeddings['cup'][:, 0],
        y=embedding_info[:,1],#part_embeddings['handle'][:, 1],
    )
    fig.update_traces(marker_color="rgba(0,0,0,0)")

    # x_range = np.max(part_embeddings['cup'][:, 0]) - np.min(part_embeddings['cup'][:, 0])
    # y_range = np.max(part_embeddings['handle'][:, 0]) - np.min(part_embeddings['handle'][:, 0])
    x_range = np.max(embedding_info[:,0]) - np.min(embedding_info[:,0])
    y_range = np.max(embedding_info[:,1]) - np.min(embedding_info[:,1])

    for i in range(len(pcl_ids)):
        pcl_id, cost = pcl_ids[i], part_costs[i]
        fig.add_layout_image(
            dict(
                source=color_adjust_image(
                    circlify_image(Image.open(f"mesh_images/mugs/{pcl_id}.png")),
                    cost,
                    cost_thresh,
                    pcl_ids[i], training_ids,
                ),
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                x=embedding_info[i,0],#part_embeddings['cup'][i, 0],
                y=embedding_info[i,1],#part_embeddings['handle'][i, 0],
                sizex=x_range / 2.5,
                sizey=y_range / 2.5,
                sizing="contain",
                #opacity=0.8,
                layer="above",
            )
        )
    fig.update_layout(
        title=dict(text="Part Decomposition Shape Warping",  x=0.5,  font=dict(size=25)),
    )
    fig.update_layout(xaxis=dict(
        title=dict(
            text="v1",
            font=dict(size=25),
        )
    ),)

    fig.update_layout(yaxis=dict(
        title=dict(
            text="v2",
            font=dict(size=25)
        )
    ),)

    fig.update_xaxes(range=[-2.5, 1])
    fig.update_yaxes(range=[-2.6, -0.8])
    fig.write_image("part_costs.png")
    fig.show()


def get_whole_cost_graph(object_embeddings, object_costs, cost_thresh, pcl_ids, training_ids):
    # object_embeddings = TSNE(
    #     n_components=2, learning_rate="auto", init="random", perplexity=3
    # ).fit_transform(object_warps)
    fig = px.scatter(
        x=object_embeddings[:, 0],
        y=object_embeddings[:, 1],
    )
    fig.update_traces(marker_color="rgba(0,0,0,0)")
    x_range = np.max(object_embeddings[:, 0]) - np.min(object_embeddings[:, 0])
    y_range = np.max(object_embeddings[:, 0]) - np.min(object_embeddings[:, 0])

    for i in range(len(pcl_ids)):
        pcl_id, embedding, cost = pcl_ids[i], object_embeddings[i], object_costs[i]
        fig.add_layout_image(
            dict(
                source=color_adjust_image(
                    circlify_image(Image.open(f"mesh_images/mugs/{pcl_id}.png")),
                    cost,
                    cost_thresh,
                    pcl_ids[i], training_ids,
                ),
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                x=embedding[0],
                y=embedding[1],
                sizex=x_range / 4,
                sizey=y_range / 4,
                sizing="contain",
                layer="above",
            )
        )

    fig.update_layout(
        title=dict(text="Whole Object Shape Warping",  x=0.5,  font=dict(size=25)),
    )
    fig.update_xaxes(range=[-2.5, 1])
    fig.update_yaxes(range=[-2.6, -0.8])
    fig.write_image("whole_costs.png")
    fig.update_layout(xaxis=dict(
        title=dict(
            text="v1",
            font=dict(size=25),
        )
    ),)

    fig.update_layout(yaxis=dict(
        title=dict(
            text="v2",
            font=dict(size=25)
        )
    ),)
    fig.write_image("whole_costs.png")

    fig.show()
    
def get_rndf_cost_graph(object_embeddings, training_object_embeddings, object_costs, cost_thresh, pcl_ids, training_ids):
    # object_embeddings = TSNE(
    #     n_components=2, learning_rate="auto", init="random", perplexity=3
    # ).fit_transform(object_warps)
    fig = px.scatter(
        x=object_embeddings[:, 0],
        y=object_embeddings[:, 1],
    )
    fig.update_traces(marker_color="rgba(0,0,0,0)")
    x_range = np.max(object_embeddings[:, 0]) - np.min(object_embeddings[:, 0])
    y_range = np.max(object_embeddings[:, 0]) - np.min(object_embeddings[:, 0])

    for i in range(len(training_ids)):
        embedding = training_object_embeddings[i]
        fig.add_layout_image(
            dict(
                source=color_adjust_image(
                    circlify_image(Image.open(f"mesh_images/mugs/{training_ids[i]}.png")),
                    0,
                    .5,
                    training_ids[i], training_ids,
                ),
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                x=embedding[0],
                y=embedding[1],
                sizex=x_range / 2,
                sizey=y_range / 2,
                sizing="contain",
                layer="above",
            )
        )



    for i in range(len(pcl_ids)):
        pcl_id, embedding, cost = pcl_ids[i], object_embeddings[i], object_costs[i]
        fig.add_layout_image(
            dict(
                source=color_adjust_image(
                    circlify_image(Image.open(f"mesh_images/mugs/{pcl_id}.png")),
                    cost,
                    cost_thresh,
                    pcl_ids[i], training_ids,
                ),
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                x=embedding[0],
                y=embedding[1],
                sizex=x_range / 2,
                sizey=y_range / 2,
                sizing="contain",
                layer="above",
            )
        )

    fig.update_layout(
        title=dict(text="Relational Neural Descriptor Fields",  x=0.5, font=dict(size=25)),

        )
    fig.update_layout(xaxis=dict(
        title=dict(
            text="v1",
            font=dict(size=25),
        )
    ),)

    fig.update_layout(yaxis=dict(
        title=dict(
            text="v2",
            font=dict(size=25)
        )
    ),)

    

    fig.update_xaxes(range=[-2.5, 1])
    fig.update_yaxes(range=[-2.6, -0.8])
    fig.write_image("rndf_costs.png")
    fig.show()

def get_lndf_cost_graph(object_embeddings, training_object_embeddings, object_costs, cost_thresh, pcl_ids, training_ids):
    # object_embeddings = TSNE(
    #     n_components=2, learning_rate="auto", init="random", perplexity=3
    # ).fit_transform(object_warps)
    fig = px.scatter(
        x=object_embeddings[:, 0],
        y=object_embeddings[:, 1],
    )
    fig.update_traces(marker_color="rgba(0,0,0,0)")
    x_range = np.max(object_embeddings[:, 0]) - np.min(object_embeddings[:, 0])
    y_range = np.max(object_embeddings[:, 0]) - np.min(object_embeddings[:, 0])

    for i in range(len(training_ids)):
        embedding = training_object_embeddings[i]
        fig.add_layout_image(
            dict(
                source=color_adjust_image(
                    circlify_image(Image.open(f"mesh_images/mugs/{training_ids[i]}.png")),
                    0,
                    .5,
                    training_ids[i], training_ids,
                ),
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                x=embedding[0],
                y=embedding[1],
                sizex=x_range/2.5 ,
                sizey=y_range/2.5 ,
                sizing="contain",
                layer="above",
            )
        )


    for i in range(len(pcl_ids)):
        pcl_id, embedding, cost = pcl_ids[i], object_embeddings[i], object_costs[i]
        fig.add_layout_image(
            dict(
                source=color_adjust_image(
                    circlify_image(Image.open(f"mesh_images/mugs/{pcl_id}.png")),
                    cost,
                    cost_thresh,
                    pcl_ids[i], training_ids,
                ),
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                x=embedding[0],
                y=embedding[1],
                sizex=x_range/2.5 ,
                sizey=y_range/2.5 ,
                sizing="contain",
                layer="above",
            )
        )
    
    fig.update_layout(
        title=dict(text="Local Neural Descriptor Fields",  x=0.5,  font=dict(size=25)),
    )
    fig.update_layout(xaxis=dict(
        title=dict(
            text="v1",
            font=dict(size=25),
        )
    ),)

    fig.update_layout(yaxis=dict(
        title=dict(
            text="v2",
            font=dict(size=25)
        )
    ),)


    fig.update_xaxes(range=[-2.5, 1])
    fig.update_yaxes(range=[-2.6, -0.8])
    fig.write_image("lndf_costs.png")
    fig.show()
        
def get_experiment_results():
    results_dict = {}
    whole_results_dict = {}
    rndf_results_dict = {}
    lndf_results_dict = {}
    
    exp_type = 'mug_on_rack'#'bowl_on_mug'#
    root_dir = f'/home/rthomp12/relational_ndf/src/rndf_robot/eval_data/eval_data/intersect_test_exp--{exp_type}_sweep_demo-exp--release_demos/'

    #root_dir = f'/home/rthomp12/relational_ndf/src/rndf_robot/eval_data/eval_data/exp--{exp_type}_upright_pose_new_demo-exp--release_demos/'
    experiment_folders = os.listdir(root_dir) 
    rndf_folders = [folder for folder in experiment_folders if 'rndf' in folder]
    lndf_folders = [folder for folder in experiment_folders if 'lndf' in folder]
    part_whole_folders = [folder for folder in experiment_folders if 'rndf' not in folder and 'lndf' not in folder and folder != 'old']
    
    #part and whole results
    for exp_folder in part_whole_folders:
        # print(exp_folder)
        # objects_raw = os.listdir(root_dir+exp_folder) 

        exp_folder += '/'# + objects_raw[0] + '/'

        objects_raw = os.listdir(root_dir+exp_folder) 
        trial_folders = [fn for fn in objects_raw if (fn.split('_')[0] == 'trial')]

        for folder in trial_folders:
            #print(folder)
            try:
                experiment_file = root_dir + exp_folder + folder + '/parts_based_success_rate_relation.npz'
                result = np.load(experiment_file, allow_pickle=True)
                
                child_id = str(result['child_id'])
                if child_id in results_dict.keys():
                    results_dict[child_id].append(1 if result['place_success'] else 0)
                else:
                    results_dict[child_id] = [1] if result['place_success'] else [0]
            except FileNotFoundError:
                continue
                
            try:
                experiment_file = root_dir + exp_folder + folder + '/whole_success_rate_relation.npz'
                result = np.load(experiment_file, allow_pickle=True)
                
                child_id = str(result['child_id'])
                if child_id in whole_results_dict.keys():
                    whole_results_dict[child_id].append(1 if result['place_success'] else 0)
                else:
                    whole_results_dict[child_id] = [1] if result['place_success'] else [0]
            except FileNotFoundError:
                continue

    results_nums = {id_number:sum(results_dict[id_number])/len(results_dict[id_number]) for id_number in results_dict.keys()}
    whole_nums = {id_number:sum(whole_results_dict[id_number])/len(whole_results_dict[id_number]) for id_number in whole_results_dict.keys()}

    for exp_folder in rndf_folders:
        exp_folder += '/'
        exp_folder += os.listdir(root_dir+exp_folder)[0]
        exp_folder += '/'
        objects_raw = os.listdir(root_dir+exp_folder) 
        trial_folders = [fn for fn in objects_raw if (fn.split('_')[0] == 'trial')]
        for folder in trial_folders:
            #print(folder)
            try:
                experiment_file = root_dir + exp_folder + folder + '/success_rate_relation.npz'
                result = np.load(experiment_file, allow_pickle=True)
                
                child_id = str(result['child_id'])
                if child_id in rndf_results_dict.keys():
                    rndf_results_dict[child_id].append(1 if result['place_success'] else 0)
                else:
                    rndf_results_dict[child_id] = [1] if result['place_success'] else [0]
            except FileNotFoundError:
                continue

    for exp_folder in lndf_folders:
        exp_folder += '/'
        objects_raw = os.listdir(root_dir+exp_folder) 
        trial_folders = [fn for fn in objects_raw if (fn.split('_')[0] == 'trial')]
        for folder in trial_folders:
            #print(folder)
            try:
                experiment_file = root_dir + exp_folder + folder + '/success_rate_relation.npz'
                result = np.load(experiment_file, allow_pickle=True)
                
                child_id = str(result['child_id'])
                if child_id in lndf_results_dict.keys():
                    lndf_results_dict[child_id].append(1 if result['place_success'] else 0)
                else:
                    lndf_results_dict[child_id] = [1] if result['place_success'] else [0]
            except FileNotFoundError:
                continue

    rndf_nums = {id_number:sum(rndf_results_dict[id_number])/len(rndf_results_dict[id_number]) for id_number in rndf_results_dict.keys()}
    lndf_nums = {id_number:sum(lndf_results_dict[id_number])/len(lndf_results_dict[id_number]) for id_number in lndf_results_dict.keys()}
    return results_nums, whole_nums, rndf_nums, lndf_nums



cfg = get_eval_cfg_defaults()
config_fname = osp.join(
    path_util.get_rndf_config(), "eval_cfgs", "base_cfg"
)  # args.config)
if osp.exists(config_fname):
    cfg.merge_from_file(config_fname)
else:
    log_info(f"Config file {config_fname} does not exist, using defaults")

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

    segmentable_filtered = []
    for obj in objects_filtered:
        if check_segmentation_exists(obj):
            segmentable_filtered.append(obj)

    mesh_names[k] = segmentable_filtered  # objects_filtered

# get_syn_rack_images(mesh_names.keys())

warp_file_stamp = "20240320-032402"

# todo: generalize for other objects
object_warp_file = f"./part_based_warp_models/old/whole_mug_{warp_file_stamp}"
cup_warp_file = f"./part_based_warp_models/old/cup_{warp_file_stamp}"
handle_warp_file = f"./part_based_warp_models/old/handle_{warp_file_stamp}"

part_names = ["cup", "handle"]
part_labels = {"cup": 37, "handle": 36}

part_canonicals = {}
whole_object_canonical = pickle.load(open(object_warp_file, "rb"))
part_canonicals["cup"] = pickle.load(open(cup_warp_file, "rb"))
part_canonicals["handle"] = pickle.load(open(handle_warp_file, "rb"))

obj_classes = list(mesh_names.keys())

scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
scale_default = cfg.MESH_SCALE_DEFAULT

# cfg.OBJ_SAMPLE_Y_HIGH_LOW = [0.3, -0.3]
cfg.OBJ_SAMPLE_Y_HIGH_LOW = [-0.35, 0.175]
x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
table_z = cfg.TABLE_Z

obj_type = "mug"
# get_object_images(mesh_names[obj_type])

# get_all_object_warps(mesh_names[obj_type])
#get_all_part_warps(mesh_names[obj_type], part_names)

#get_whole_reconstructions(whole_object_canonical, part_names, warp_file_stamp, mesh_names[obj_type])
# get_part_reconstructions(part_canonicals, part_names, warp_file_stamp, mesh_names[obj_type])

# whole_costs = np.load(f"whole_mug_costs_resampled_{warp_file_stamp}.npy")
# cost_thresh = np.mean(whole_costs)
# print(cost_thresh)
# whole_costs = np.load(f"whole_mug_costs_{warp_file_stamp}.npy")
# cost_thresh = np.mean(whole_costs)
# print(cost_thresh)

# cup_costs = np.load(f"cup_costs_{warp_file_stamp}.npy")
# handle_costs = np.load(f"handle_costs_{warp_file_stamp}.npy")
# part_costs = (cup_costs + handle_costs) / 2

result_nums, whole_nums, rndf_nums, lndf_nums = get_experiment_results()

whole_params = pickle.load(open(f'whole_mug_params_resampled_{warp_file_stamp}', 'rb'))
cup_params = pickle.load(open(f'cup_params_{warp_file_stamp}', 'rb'))
part_params = pickle.load(open(f'handle_params_{warp_file_stamp}', 'rb'))

#whole_embeddings = np.array([param.latents for param in whole_params])
cup_embeddings = np.array([param.latent for param in part_params['cup']])
handle_embeddings = np.array([param.latent for param in part_params['handle']])


# print(cost_thresh)
# print(part_costs)

warps = np.load("all_mug_warps.pkl.npy")
part_warps = {}
for part in part_names:
    part_warps[part] = np.load(f"all_{part}_warps.pkl.npy")

whole_embeddings = PCA(n_components=2).fit_transform(warps)
part_embeddings = {}
# for part in part_names:
#     part_embeddings[part] = PCA(n_components=2).fit_transform(part_warps[part])
part_embeddings['cup'] = cup_embeddings
part_embeddings['handle'] = handle_embeddings

filtered_whole_mesh_names = [mesh_name for mesh_name in mesh_names[obj_type] if mesh_name in whole_nums.keys()]
filtered_whole_mesh_idxs = [i for i, mesh_name in enumerate(mesh_names[obj_type]) if mesh_name in whole_nums.keys()]

filtered_whole_embeddings = whole_embeddings[filtered_whole_mesh_idxs]
fake_whole_costs = [whole_nums[mesh_name] for mesh_name in filtered_whole_mesh_names]


filtered_mesh_names = [mesh_name for mesh_name in mesh_names[obj_type] if mesh_name in result_nums.keys()]
filtered_mesh_idxs = [i for i, mesh_name in enumerate(mesh_names[obj_type]) if mesh_name in result_nums.keys()]
filtered_cup_embeddings = part_embeddings['cup'][filtered_mesh_idxs]
filtered_handle_embeddings = part_embeddings['handle'][filtered_mesh_idxs]
fake_mesh_costs = [result_nums[mesh_name] for mesh_name in filtered_mesh_names]

whole_part_embeddings = {'cup': filtered_cup_embeddings, 'handle': filtered_handle_embeddings}
training_ids = whole_object_canonical.metadata.training_ids


filtered_rndf_mesh_names = [mesh_name for mesh_name in mesh_names[obj_type] if mesh_name in rndf_nums.keys()]
filtered_rndf_mesh_idxs = [i for i, mesh_name in enumerate(mesh_names[obj_type]) if mesh_name in rndf_nums.keys()]
rndf_costs = [rndf_nums[mesh_name] for mesh_name in filtered_rndf_mesh_names]
rndf_part_embeddings = {'cup': part_embeddings['cup'][filtered_rndf_mesh_idxs], 'handle': part_embeddings['handle'][filtered_rndf_mesh_idxs]}
rndf_training_ids  = np.loadtxt('./scripts/mug_train_object_split.txt', dtype=str)
filtered_rndf_training_ids = [mesh_name for mesh_name in mesh_names[obj_type] if mesh_name in rndf_training_ids]
rndf_training_part_idxs =  [i for i, mesh_name in enumerate(mesh_names[obj_type]) if mesh_name in rndf_training_ids]
rndf_training_part_embeddings = {'cup': part_embeddings['cup'][rndf_training_part_idxs], 'handle': part_embeddings['handle'][rndf_training_part_idxs]}

filtered_lndf_mesh_names = [mesh_name for mesh_name in mesh_names[obj_type] if mesh_name in lndf_nums.keys()]
filtered_lndf_mesh_idxs = [i for i, mesh_name in enumerate(mesh_names[obj_type]) if mesh_name in lndf_nums.keys()]
lndf_costs = [lndf_nums[mesh_name] for mesh_name in filtered_lndf_mesh_names]
lndf_part_embeddings = {'cup': part_embeddings['cup'][filtered_lndf_mesh_idxs], 'handle': part_embeddings['handle'][filtered_lndf_mesh_idxs]}
lndf_training_ids  = np.loadtxt('./scripts/lndf_mug_train_object_split.txt', dtype=str)
filtered_lndf_training_ids = [mesh_name for mesh_name in mesh_names[obj_type] if mesh_name in lndf_training_ids]
lndf_training_part_idxs =  [i for i, mesh_name in enumerate(mesh_names[obj_type]) if mesh_name in lndf_training_ids]
lndf_training_part_embeddings = {'cup': part_embeddings['cup'][lndf_training_part_idxs], 'handle': part_embeddings['handle'][lndf_training_part_idxs]}

print(rndf_training_ids)

print(whole_nums)
get_whole_cost_graph(np.concatenate([np.atleast_2d(whole_part_embeddings['cup'][:, 0]).T, np.atleast_2d(whole_part_embeddings['handle'][:,0]).T], axis=1), fake_whole_costs, .8, filtered_whole_mesh_names, training_ids)
get_part_cost_graph(whole_part_embeddings, fake_mesh_costs, .8, filtered_mesh_names, training_ids)
get_rndf_cost_graph(np.concatenate([np.atleast_2d(rndf_part_embeddings['cup'][:, 0]).T, np.atleast_2d(rndf_part_embeddings['handle'][:,0]).T], axis=1), 
                                    np.concatenate([np.atleast_2d(rndf_training_part_embeddings['cup'][:, 0]).T, np.atleast_2d(rndf_training_part_embeddings['handle'][:,0]).T], axis=1),
                                    rndf_costs, .8, filtered_rndf_mesh_names, filtered_rndf_training_ids)
get_lndf_cost_graph(np.concatenate([np.atleast_2d(lndf_part_embeddings['cup'][:, 0]).T, np.atleast_2d(lndf_part_embeddings['handle'][:,0]).T], axis=1), 
                                    np.concatenate([np.atleast_2d(lndf_training_part_embeddings['cup'][:, 0]).T, np.atleast_2d(lndf_training_part_embeddings['handle'][:,0]).T], axis=1),
                                    lndf_costs, .8, filtered_lndf_mesh_names, filtered_lndf_training_ids)


warp_ids = [43,98,103,91,8,88,49,100,96,97]

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

rndf_image = Image.open('rndf_costs.png') 
lndf_image = Image.open('lndf_costs.png') 
whole_image = Image.open('whole_costs.png') 
part_image = Image.open('part_costs.png') 

final_figure = image_grid([rndf_image, lndf_image, whole_image, part_image], 2, 2)
final_figure.save('graph_figure.png')

# get_object_embeddings(whole_embeddings, mesh_names[obj_type])
# get_part_embeddings(part_embeddings, part_names, mesh_names[obj_type])
