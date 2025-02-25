import os, os.path as osp
from src import utils, viz_utils
from src.utils import CanonPart, CanonPartMetadata, pos_quat_to_transform, ObjParam
import trimesh
import numpy as np
import pickle
from scipy.spatial.transform import Rotation
import copy as cp
from dataclasses import dataclass

from sklearn import neighbors
from sklearn.decomposition import PCA
from numpy.typing import NDArray
from typing import List, Optional, Tuple, Union
from rndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from rndf_robot.utils import util, path_util
from airobot import log_info
from airobot.utils import common
from rndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
from scripts.mug_utils import load_all_shapenet_files, load_segmented_pointcloud_from_txt, get_mesh, get_segmented_mesh

#returns the ids of a sampled set of objects for training
def sample_training(num_objects, obj_type, directory):
    try:
        print('%s_test_object_split.txt' % obj_type)
        test_ids = np.loadtxt(osp.join(path_util.get_rndf_share(), '%s_test_object_split.txt' % obj_type), dtype=str).tolist()
        test_ids = [val.split('.')[0] for val in test_ids] 
    except FileNotFoundError:
        print("Test obj file not found")
        test_ids = []

    all_shapenet_mugs = load_all_shapenet_files(obj_type)

    training_ids = []
    while len(training_ids) < num_training:
        candidate_id = all_shapenet_mugs[np.random.choice(len(all_shapenet_mugs))]
        if obj_type == 'mug': 
            if candidate_id not in training_ids and candidate_id not in test_ids:
                try:
                    load_segmented_pointcloud_from_txt(candidate_id)
                    training_ids.append(candidate_id)
                except:
                    continue
        else:
            training_ids.append(candidate_id)
    return training_ids

def pick_canonical(decomposed_meshes, part_names, num_surface_samples=10000):
    small_surface_points = []
    surface_points = []
    mesh_points = []
    hybrid_points = []
    faces = []
    centers = []

    part_small_surface_points = {part:[] for part in part_names}
    part_surface_points  = {part:[] for part in part_names}
    part_mesh_points = {part:[] for part in part_names}
    part_hybrid_points = {part:[] for part in part_names}
    part_faces = {part:[] for part in part_names}
    part_centers = {part:[] for part in part_names}

    #get canonical from considering the whole meshes (concatenated from parts)
    reconfigured_meshes = [{} for _ in range(len(decomposed_meshes[part_names[0]]))]
    for part in part_names: 
        for i in range(len(decomposed_meshes[part])):
            reconfigured_meshes[i][part] = decomposed_meshes[part][i]


    for decomp_mesh in reconfigured_meshes:
        for part in part_names: 
            mesh = decomp_mesh[part]
            
            #utils.trimesh_transform(mesh, center=False, rotation=rotation)
            part_t = mesh.centroid

            part_sp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=num_surface_samples)
            part_ssp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
            part_mp, part_f = utils.trimesh_get_vertices_and_faces(mesh)

            part_small_surface_points[part].append(part_ssp)
            part_surface_points[part].append(part_sp)
            part_mesh_points[part].append(part_mp)
            part_faces[part].append(part_f)
            part_centers[part].append(part_t)

        scaled_part_points = utils.scale_points_circle(sum([[part_small_surface_points[part][-1], 
                                                             part_surface_points[part][-1], 
                                                             part_mesh_points[part][-1], 
                                                             np.atleast_2d(part_centers[part][-1])] for part in part_names], []), base_scale=0.1)
        for part in part_names:
            part_small_surface_points[part][-1] = scaled_part_points.pop(0)
            part_surface_points[part][-1] = scaled_part_points.pop(0)
            part_mesh_points[part][-1] = scaled_part_points.pop(0)
            part_centers[part][-1] = scaled_part_points.pop(0)
        assert len(scaled_part_points) == 0

        for part in part_names: 
            part_h = np.concatenate([part_mesh_points[part][-1], 
                                     part_surface_points[part][-1]])  # Order important!
            part_hybrid_points[part].append(part_h)

        whole_mesh = trimesh.util.concatenate([decomp_mesh[part] for part in part_names])
        viz_utils.show_meshes_plotly({'whole': whole_mesh.vertices}, {'whole': whole_mesh.faces})
        ssp = np.concatenate([part_small_surface_points[part][-1] for part in part_names])
        sp = np.concatenate([part_surface_points[part][-1] for part in part_names])
        mp = np.concatenate([part_mesh_points[part][-1] for part in part_names])
        whole_faces = []
        num_mp = 0
        for part in part_names:
            whole_faces.append(part_faces[part][-1]+num_mp)
            num_mp += len(part_mesh_points[part][-1])
        f = np.concatenate(whole_faces)
        h = np.concatenate([mp, sp])  # Order important!
        # viz_utils.show_pcds_plotly({'mp': mp, 'sp': sp, 'hybrid':h})
        # input("continue?")

        t = whole_mesh.centroid

        #translation_matrix[:3, 3] = t.squeeze()
        centers.append(t)
        small_surface_points.append(ssp)
        surface_points.append(sp)
        mesh_points.append(mp)
        faces.append(f)
        hybrid_points.append(h)

    whole_objects = {'centers': centers,
                     'small_surface_points': small_surface_points,
                     'surface_points': surface_points,
                     'mesh_points': mesh_points,
                     'faces': faces,
                     'hybrid_points': hybrid_points}
    decomposed_objects = {'centers': part_centers,
                         'small_surface_points': part_small_surface_points,
                         'surface_points': part_surface_points,
                         'mesh_points': part_mesh_points,
                         'faces': part_faces,
                         'hybrid_points': part_hybrid_points}

    canonical_idx = utils.sst_pick_canonical(hybrid_points)
    return canonical_idx, whole_objects, decomposed_objects


def learn_warps(meshes, n_dimensions, canonical_idx = None, num_surface_samples=10000):
    #rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()
    rotation = Rotation.from_euler("zyx", [0., 0., 0]).as_matrix()

    small_surface_points = []
    surface_points = []
    mesh_points = []
    hybrid_points = []
    faces = []
    centers = []

    for mesh in meshes:
        translation_matrix = np.eye(4)
        utils.trimesh_transform(mesh, center=False, rotation=rotation)
        t = mesh.centroid

        sp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=num_surface_samples)
        ssp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
        mp, f = utils.trimesh_get_vertices_and_faces(mesh)
        ssp, sp, mp, t = utils.scale_points_circle([ssp, sp, mp, np.atleast_2d(t)], base_scale=0.1)
        h = np.concatenate([mp, sp])  # Order important!
        translation_matrix[:3, 3] = t.squeeze()
        centers.append(t)
        small_surface_points.append(ssp)
        surface_points.append(sp)
        mesh_points.append(mp)
        faces.append(f)
        hybrid_points.append(h)

    if canonical_idx is None:
        canonical_idx = utils.sst_pick_canonical(hybrid_points)

    print(f"Canonical obj index: {canonical_idx}.")

    tmp_obj_points = cp.copy(small_surface_points)
    tmp_obj_points[canonical_idx] = hybrid_points[canonical_idx]

    


    warps, _ = utils.warp_gen(canonical_idx, tmp_obj_points, alpha=0.01, visualize=True)
    # pickle.dump(warps, open('temp_data', 'wb'))
    #warps = pickle.load(open('temp_data', 'rb'))
    #_, pca = utils.pca_transform(warps, n_dimensions=n_dimensions)
    _, pca = utils.ae_transform(warps, n_dimensions=n_dimensions)


    warp_results = {
            "pca": pca,
            'canonical_idx': canonical_idx,
            "canonical_pcl": hybrid_points[canonical_idx],
            "canonical_mesh_points": mesh_points[canonical_idx],
            "canonical_mesh_faces": faces[canonical_idx], 
            "canonical_center_transform": centers[canonical_idx]
        }
    
    return warp_results


def learn_concat_warps(decomposed_meshes, part_names, n_dimensions, canonical_idx = None, num_surface_samples=10000):
    
    canonical_idx, whole_objects, decomposed_objects = pick_canonical(decomposed_meshes, part_names)
    print(f"Canonical obj index: {canonical_idx}.")


    #check that the canonicals have the correct unumber of points
    
    # exit(0)

    part_warps = {}
    for part in part_names:
        small_surface_points = decomposed_objects['small_surface_points'][part]
        tmp_obj_points = cp.deepcopy(small_surface_points)
        tmp_obj_points[canonical_idx] = cp.deepcopy(decomposed_objects['hybrid_points'][part][canonical_idx])
        part_warps[part], _ = utils.warp_gen(canonical_idx, tmp_obj_points, alpha=0.01, visualize=True)
        part_warps[part] = cp.deepcopy(part_warps[part])


    print(whole_objects['hybrid_points'][canonical_idx].shape)
    for part in part_names:
        print(part)
        #print(decomposed_objects['hybrid_points'][part][canonical_idx].shape)
        print(decomposed_objects['mesh_points'][part][canonical_idx].shape)
        print(decomposed_objects['surface_points'][part][canonical_idx].shape)
        print(decomposed_objects['hybrid_points'][part][canonical_idx].shape)
        print()

    # for part in part_names:
    #     print(part)
    #     print(part_warps[part][0].shape)
    #     print(part_warps[part][0].reshape(-1,3))
    # print()


    #reassemble
    final_warp_mats = [[] for i in range(len(whole_objects['centers'])-1)]
    for i in range(len(final_warp_mats)):
        for part in part_names:

            final_warp_mats[i].append(part_warps[part][i].reshape(-1,3)[:len(decomposed_objects['mesh_points'][part][canonical_idx])])
        for part in part_names:
            final_warp_mats[i].append(part_warps[part][i].reshape(-1,3)[len(decomposed_objects['mesh_points'][part][canonical_idx]):])

        for mat in final_warp_mats[i]:
            print(mat.shape)
        final_warp_mats[i] = np.concatenate(final_warp_mats[i])
        final_warp_mats[i] = np.hstack(final_warp_mats[i])

    # for mat in final_warp_mats:
    #     viz_utils.show_pcds_plotly({'canon': whole_objects['hybrid_points'][canonical_idx],
    #                                 'warped': whole_objects['hybrid_points'][canonical_idx] + mat}).show()

    #_, pca = utils.pca_transform(final_warp_mats, n_dimensions=n_dimensions)
    _, pca = utils.ae_transform(final_warp_mats, n_dimensions=n_dimensions)

    warp_results = {
            "pca": pca,
            'canonical_idx': canonical_idx,
            "canonical_pcl": whole_objects['hybrid_points'][canonical_idx],
            "canonical_mesh_points": whole_objects['mesh_points'][canonical_idx],
            "canonical_mesh_faces": whole_objects['faces'][canonical_idx], 
            "canonical_center_transform": whole_objects['centers'][canonical_idx]
        }
    
    return warp_results

if __name__ == "__main__":
    print("Generating Warps")

    num_training = 10
    n_dimensions = 8

    obj_type = 'mug'
    part_labels = {'cup': 37, 'handle': 36}
    part_names = ['cup', 'handle']
    directory = "../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/"

    training_ids = sample_training(num_training, obj_type, directory)
    training_whole_meshes = []
    training_part_meshes = {part:[] for part in part_names}

    if obj_type == "mug":   
        for obj_id in training_ids:
            training_whole_meshes.append(get_mesh(obj_id))
            part_meshes = get_segmented_mesh(obj_id)
            for part in part_names: 
                training_part_meshes[part].append(part_meshes[part_labels[part]])

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    print("Part warping")
    part_warps = {}

    #Whole mug warps
    whole_obj_warp_name = f"ae_whole_{obj_type}_{timestr}_{len(training_ids)}"
    whole_obj_tag = 'none'


    whole_obj_warp_data = learn_warps(training_whole_meshes, n_dimensions=n_dimensions)
    whole_obj_metadata = CanonPartMetadata(whole_obj_tag,
                                            training_ids[whole_obj_warp_data['canonical_idx']],
                                            training_ids[:whole_obj_warp_data['canonical_idx']] + training_ids[whole_obj_warp_data['canonical_idx']+1:],
                                            part_label=None)

    whole_obj_warps = CanonPart(whole_obj_warp_data['canonical_pcl'], 
                                whole_obj_warp_data['canonical_mesh_points'],
                                whole_obj_warp_data['canonical_mesh_faces'],
                                whole_obj_warp_data['canonical_center_transform'],                                 
                                contact_points=None,
                                metadata=whole_obj_metadata,
                                pca=whole_obj_warp_data['pca'],)

    pickle.dump(whole_obj_warps, open(whole_obj_warp_name, 'wb'))


    part_warp_data = learn_concat_warps(training_part_meshes, part_names, n_dimensions=n_dimensions,)

    part_warp_name = f"ae_{obj_type}_concat_{timestr}_{len(training_ids)}"
    part_tag = 'none'
    part_metadata = CanonPartMetadata(part_tag,
                                        training_ids[part_warp_data['canonical_idx']],
                                        training_ids[:part_warp_data['canonical_idx']] + training_ids[part_warp_data['canonical_idx']+1:],
                                        part_label=None)

    part_warps[part] = CanonPart(part_warp_data['canonical_pcl'], 
                           part_warp_data['canonical_mesh_points'],
                           part_warp_data['canonical_mesh_faces'],
                            part_warp_data['canonical_center_transform'],                                 
                            contact_points=None,
                            metadata=part_metadata,
                            pca=part_warp_data['pca'],)
    pickle.dump(part_warps[part], open(part_warp_name, 'wb'))




