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

#Define the adjusted canonical object class here? 
#It should behave like the one in utils 
#but should also keep the names of the objects it was trained on 
#and should also keep ahold of the contact points 
#and any information about the parameters 
#and a name for the kind of warp run

#names of the kinds of warps we might have: 
#whole mug 
#cup
#handle
#prealigned

NPF32 = NDArray[np.float32]
NPF64 = NDArray[np.float64]

# @dataclass
# class CanonPartMetadata:
#     experiment_tag: str 
#     canonical_id: str
#     training_ids: List[str]
#     part_label: Optional[int]

# @dataclass
# class CanonPart:
#     """Canonical object with shape warping."""
#     canonical_pcl: NPF32
#     mesh_vertices: NPF32
#     mesh_faces: NDArray[np.int32]
#     center_transform: NPF32 #For saving the relative transform of parts
#     metadata: CanonPartMetadata
#     contact_points: Optional[List[int]]
#     pca: Optional[PCA] = None

#     def __post_init__(self):
#         if self.pca is not None:
#             self.n_components = self.pca.n_components

#     def to_pcd(self, obj_param: ObjParam) -> NPF32:
#         if self.pca is not None and obj_param.latent is not None:
#             pcd = self.canonical_pcl + self.pca.inverse_transform(obj_param.latent).reshape(-1, 3)
#         else:
#             if self.pca is not None:
#                 print("WARNING: Skipping warping because we do not have a latent vector. We however have PCA.")
#             pcd = np.copy(self.canonical_pcl)
#         return pcd * obj_param.scale[None]

#     def to_transformed_pcd(self, obj_param: ObjParam) -> NPF32:
#         pcd = self.to_pcd(obj_param)
#         trans = utils.pos_quat_to_transform(obj_param.position, obj_param.quat)
#         return utils.transform_pcd(pcd, trans)

#     def to_mesh(self, obj_param: ObjParam) -> trimesh.Trimesh:
#         pcd = self.to_pcd(obj_param)
#         # The vertices are assumed to be at the start of the canonical_pcd.
#         vertices = pcd[:len(self.mesh_vertices)]
#         return trimesh.Trimesh(vertices, self.mesh_faces)

#     def to_transformed_mesh(self, obj_param: ObjParam) -> trimesh.Trimesh:
#         pcd = self.to_transformed_pcd(obj_param)
#         # The vertices are assumed to be at the start of the canonical_pcd.
#         vertices = pcd[:len(self.mesh_vertices)]
#         return trimesh.Trimesh(vertices, self.mesh_faces)

#     @staticmethod
#     def from_pickle(load_path: str) -> "CanonObj":
#         with open(load_path, "rb") as f:
#             data = pickle.load(f)
#         pcd = data["canonical_obj"]
#         pca = None
#         center_transform = np.eye(4)
#         if 'center_transform' in data:
#             center_transform = data['center_transform']
#         if "pca" in data:
#             pca = data["pca"]
#         mesh_vertices = data["canonical_mesh_points"]
#         mesh_faces = data["canonical_mesh_faces"]
#         return CanonObj(pcd, mesh_vertices, mesh_faces, center_transform, pca)

def load_all_shapenet_files(obj_type):
    cfg = get_eval_cfg_defaults()
    config_fname = osp.join(path_util.get_rndf_config(), 'eval_cfgs', 'base_cfg')#args.config)
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        log_info(f'Config file {config_fname} does not exist, using defaults')

    mesh_data_dirs = {
        'mug': 'mug_centered_obj_normalized',
        # 'bottle': 'bottle_centered_obj_normalized',
        'bowl': 'bowl_centered_obj_normalized',
        'syn_rack_easy': 'syn_racks_easy_obj',
        # 'syn_container': 'box_containers_unnormalized'
    }
    mesh_data_dirs = {k: osp.join(path_util.get_rndf_obj_descriptions(), v) for k, v in mesh_data_dirs.items()}
    bad_ids = {
        'syn_rack_easy': [],
        'bowl': bad_shapenet_bowls_ids_list,
        'mug': bad_shapenet_mug_ids_list,
        'bottle': bad_shapenet_bottles_ids_list,
        'syn_container': []
    }

    upright_orientation_dict = {
        'mug': common.euler2quat([np.pi/2, 0, 0]).tolist(),
        'bottle': common.euler2quat([np.pi/2, 0, 0]).tolist(),
        'bowl': common.euler2quat([np.pi/2, 0, 0]).tolist(),
        'syn_rack_easy': common.euler2quat([0, 0, 0]).tolist(),
        'syn_container': common.euler2quat([0, 0, 0]).tolist(),
    }

    print("Iterating mesh data")
    mesh_names = {}
    for k, v in mesh_data_dirs.items():
        # get train samples
        objects_raw = os.listdir(v) 
        objects_filtered = [fn for fn in objects_raw if (fn.split('/')[-1] not in bad_ids[k] and '_dec' not in fn)]
        # objects_filtered = objects_raw
        total_filtered = len(objects_filtered)
        train_n = int(total_filtered * 0.9); test_n = total_filtered - train_n

        # train_objects = sorted(objects_filtered)[:train_n]
        # test_objects = sorted(objects_filtered)[train_n:]

        # log_info('\n\n\nTest objects: ')
        # log_info(test_objects)
        # # log_info('\n\n\n')

        mesh_names[k] = objects_filtered
    print("got em")

    obj_classes = list(mesh_names.keys())

    scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
    scale_default = cfg.MESH_SCALE_DEFAULT

    # cfg.OBJ_SAMPLE_Y_HIGH_LOW = [0.3, -0.3]
    cfg.OBJ_SAMPLE_Y_HIGH_LOW = [-0.35, 0.175]
    x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
    y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
    table_z = cfg.TABLE_Z

    return mesh_names[obj_type]

#returns the ids of a sampled set of objects for training
def sample_training(num_objects, obj_type, directory):
    print('%s_test_object_split.txt' % obj_type)
    test_ids = np.loadtxt(osp.join(path_util.get_rndf_share(), '%s_test_object_split.txt' % obj_type), dtype=str).tolist()
    test_ids = [val.split('.')[0] for val in test_ids] 

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

def load_segmented_pointcloud_from_txt(pcl_id, num_points=2048, 
                                       root = 'Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390'):
    fn = os.path.join(root, pcl_id + '.txt')
    cls = 'Mug'
    data = np.loadtxt(fn).astype(np.float32)
    point_set = data[:, 0:3]#<-- ignore the normals
    seg_ids = data[:, -1].astype(np.int32)
    point_set[:, 0:3] = utils.center_pcl(point_set[:, 0:3])

    #fixed transform to align with the other mugs being used
    rotation = Rotation.from_euler("zyx", [0., np.pi/2, 0.]).as_quat()
    transform = utils.pos_quat_to_transform([0,0,0], rotation)
    point_set = utils.transform_pcd(point_set, transform)
    #point_set = utils.scale_points_circle([point_set], base_scale=0.1)[0]
   
    return point_set, cls, seg_ids


def get_segmented_rack_mesh(rack_mesh):
    pcl = rack_mesh.vertices

    #Get the top points 
    max_z = np.max(pcl[:,2]) - .01
    max_pts = pcl[pcl[:,2] > max_z]
    center = np.mean(max_pts, 0)

    #get the furthest out point on the branch
    max_r = np.argmax(np.linalg.norm(pcl[:, :2]-center[:2], axis=-1))
    max_r_multiple = np.argpartition(np.linalg.norm(pcl[:, :2], axis=-1), -2)[-2:]

    normal = pcl[max_r,:2]/np.linalg.norm(pcl[max_r,:2])
    normal_3d = np.array([normal[0], normal[1], 0])
    normal_angle = np.arctan2(normal[1], normal[0])

    rad = max(np.linalg.norm(max_pts[:, :2]-center[:2], axis=-1))

    v = rack_mesh.vertices
    f = rack_mesh.faces

    x = np.linspace(-.5, .5, 100)

    y = normal[0]/(-normal[1])*(x-(rad*np.cos(normal_angle)+center[0])) + (rad*np.sin(normal_angle)+center[1])

    import plotly.graph_objects as go

    fig = go.Figure(
    data=[  go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=f[:, 0],
            j=f[:, 1],
            k=f[:, 2],
            colorscale="Viridis",
            intensity = v[:, 2],), go.Scatter3d(x=x, y=y, z=[.5]*100),
            go.Scatter3d(x=pcl[max_r_multiple][:,0], y=pcl[max_r_multiple][:,1], z=pcl[max_r_multiple][:,2]), 
            go.Scatter3d(x=[pcl[max_r][0]], y=[pcl[max_r][1]], z=[pcl[max_r][2]]),
            go.Scatter3d(x=max_pts[:, 0], y=max_pts[:, 1], z=max_pts[:, 2], )]
        )
    fig.show()
    #input("continue?")


    rack_trunk = trimesh.intersections.slice_mesh_plane(rack_mesh,
    -normal_3d,
    np.array([rad*np.cos(normal_angle)+center[0],rad*np.sin(normal_angle)+center[1],0]))
    #rack_trunk.show()

    rack_branch = trimesh.intersections.slice_mesh_plane(rack_mesh,
    normal_3d,
    np.array([rad*np.cos(normal_angle)+center[0],rad*np.sin(normal_angle)+center[1],0]))
    #rack_branch.show()

    
    # print("FACE COUNT")
    # print(len(rack_branch.faces))
    # print(len(rack_trunk.faces))
    # print()

    
    # rack_pts = utils.trimesh_create_verts_surface(rack_mesh, 1000)
    # trunk = rack_pts[np.linalg.norm(rack_pts[:, :2] - center[:2], axis=-1) < rad]
    # branch = rack_pts[np.linalg.norm(rack_pts[:, :2]- center[:2], axis=-1) > rad]

    return {1: rack_branch, 0:rack_trunk}

def get_rack_mesh(pcl_id):
    obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/syn_racks_easy_obj/{pcl_id}"
    mesh = utils.trimesh_load_object(obj_file_path)
    return mesh

def get_bowl(pcl_id):
    obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/bowl_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
    mesh = utils.trimesh_load_object(obj_file_path)
    return mesh


def get_segmented_mesh(pcl_id):
    seg_pcl, _, seg_ids = load_segmented_pointcloud_from_txt(pcl_id)
    obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
    unseg_mesh = utils.trimesh_load_object(obj_file_path)
    utils.trimesh_transform(unseg_mesh, center=True, scale=None, rotation=None)
    unseg_pcl, _ = utils.trimesh_get_vertices_and_faces(unseg_mesh)

    X = seg_pcl
    Y = seg_ids
    clf = neighbors.KNeighborsClassifier(2)#svm.SVC()
    clf.fit(X, Y)

    part_meshes = {}
    for part_label in clf.classes_:
        part_mesh = cp.deepcopy(unseg_mesh)
        unseg_ids = clf.predict(unseg_pcl)
        mask = unseg_ids==part_label
        face_mask = mask[part_mesh.faces].all(axis=1)
        part_mesh.update_faces(face_mask)
        part_mesh.remove_unreferenced_vertices()
        part_meshes[part_label] = part_mesh
    return part_meshes

def get_mesh(pcl_id):
    obj_file_path = f"../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{pcl_id}/models/model_normalized.obj"
    mesh = utils.trimesh_load_object(obj_file_path)
    return mesh

def get_contact_points(part_1, part_2, part_names): 
    knns = utils.get_closest_point_pairs_thresh(part_1, part_2, .00003)#get_closest_point_pairs(cup_pcl, handle_pcl, 11)
    #viz_utils.show_pcds_plotly({'cup': cup_pcl, 'handle':handle_pcl, 'pts_1': cup_knns,}) #'pts_2': handle_pcl[knns[:, 0]]})
    return {part_name[0]: knns[:,1 ], part_name[1]: knns[:, 0]}



def learn_warps(meshes, n_dimensions, num_surface_samples=10000):
    #rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()

    small_surface_points = []
    surface_points = []
    mesh_points = []
    hybrid_points = []
    faces = []
    centers = []

    for mesh in meshes:
        translation_matrix = np.eye(4)
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

    canonical_idx = utils.sst_pick_canonical(hybrid_points)
    print(f"Canonical obj index: {canonical_idx}.")

    tmp_obj_points = cp.copy(small_surface_points)
    tmp_obj_points[canonical_idx] = hybrid_points[canonical_idx]

    warps, _ = utils.warp_gen(canonical_idx, tmp_obj_points, alpha=0.01, visualize=True)
    _, pca = utils.pca_transform(warps, n_dimensions=n_dimensions)


    warp_results = {
            "pca": pca,
            'canonical_idx': canonical_idx,
            "canonical_pcl": hybrid_points[canonical_idx],
            "canonical_mesh_points": mesh_points[canonical_idx],
            "canonical_mesh_faces": faces[canonical_idx], 
            "canonical_center_transform": centers[canonical_idx]
        }
    
    return warp_results

#TODO: can set up to test the pre-alignment with whole objects, but it will require
#adjusting how the align_warp function works
#so this is incomplete/untested

def learn_warps_prealign(obj_file_paths, n_dimensions, num_surface_samples=10000):
    print("NOT IMPLEMENTED")
#     #rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()
#     centers = []
#     meshes = []

#     for obj_path in obj_file_paths:
#         mesh = utils.trimesh_load_object(obj_path)
#         translation_matrix = np.eye(4)
#         t = mesh.centroid
#         translation_matrix[:3, 3] = t
#         centers[part].append(translation_matrix)
#         utils.trimesh_transform(mesh, center=True, scale=None, rotation=None)
#         meshes.append(mesh)

#     small_surface_points = []
#     surface_points = []
#     mesh_points = []
#     hybrid_points = []
#     faces = []

#     for mesh in meshes:
#         sp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=num_surface_samples)
#         ssp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
#         mp, f = utils.trimesh_get_vertices_and_faces(mesh)
#         ssp, sp, mp = utils.scale_points_circle([ssp, sp, mp], base_scale=0.1)
#         h = np.concatenate([mp, sp])  # Order important!

#         small_surface_points.append(ssp)
#         surface_points.append(sp)
#         mesh_points.append(mp)
#         faces.append(f)
#         hybrid_points.append(h)

#     canonical_idx = utils.sst_pick_canonical(hybrid_points)
#     print(f"Canonical obj index: {canonical_idx}.")

#     tmp_obj_points = cp.copy(small_surface_points)
#     tmp_obj_points[canonical_idx] = hybrid_points[canonical_idx]

#     warps, _ = utils.warp_gen(canonical_idx, tmp_obj_points, alpha=0.01, visualize=True)
#     _, pca = utils.pca_transform(warps, n_dimensions=n_dimensions)

#     warp_results = {
#             "pca": pca,
#             "canonical_pcl": hybrid_points[canonical_idx],
#             "canonical_mesh_points": mesh_points[canonical_idx],
#             "canonical_mesh_faces": faces[canonical_idx], 
#             "canonical_center_transform": centers[canonical_idx]
#         }
    
#     return warp_results

def learn_warps_contact_prealign(meshes, 
                                 part_names, 
                                 part_labels, 
                                 n_dimensions, 
                                 num_surface_samples=10000, 
                                 n_angles=50,
                                 alpha=0.01):
    #rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()
    mesh_pcls = {}
    mesh_faces = {}
    small_pcls = {}
    hybrid_pcls = {}
    centers = {}
    canonical_idxs = {}

    small_surface_points = []
    surface_points = []
    mesh_points = []
    hybrid_points = []
    faces = []

    for mesh in meshes[part]:
        sp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=num_surface_samples)
        ssp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
        mp, f = utils.trimesh_get_vertices_and_faces(mesh)
        #ssp, sp, mp = utils.scale_points_circle([ssp, sp, mp], base_scale=0.1)
        h = np.concatenate([mp, sp])  # Order important!

        small_surface_points.append(ssp)
        surface_points.append(sp)
        mesh_points.append(mp)
        faces.append(f)
        hybrid_points.append(h)

    mesh_faces[part] = faces
    mesh_pcls[part] = mesh_points
    hybrid_pcls[part] = hybrid_points
    small_pcls[part] = small_surface_points
    canonical_idx = utils.sst_pick_canonical(hybrid_points)
    canonical_idxs[part] = canonical_idx

    #TODO： generalize to other objects
    contact_points = {} 
    for i in range(len(meshes)):
        cup_pcl = small_pcls['cup'][i]
        handle_pcl = small_pcls['handle'][i]

        if i == canonical_idxs['cup']:
            cup_pcl = hybrid_pcls['cup'][i]
        if i == canonical_idxs['handle']:
            handle_pcl = hybrid_pcls['handle'][i]

        cup_contacts, handle_contacts = get_contact_points(cup_pcl, handle_pcl)
        contact_points['cup'].append(np.array(cup_contacts))
        contact_points['handle'].append(np.array(handle_contacts))
        small_pcls['cup'][i] = utils.center_pcl(small_pcls['cup'][i])
        small_pcls['handle'][i]= utils.center_pcl(small_pcls['handle'][i])
        hybrid_pcls['cup'][i] = utils.center_pcl(hybrid_pcls['cup'][i])
        hybrid_pcls['handle'][i]= utils.center_pcl(hybrid_pcls['handle'][i])

    warp_results = {}
    for part in part_names:
        hybrid_points = hybrid_pcls[part]
        small_surface_points = small_pcls[part]
        obj_contact_points = contact_points[part]
        canonical_idx = [part]
        
        tmp_obj_points = cp.copy(small_surface_points)
        tmp_obj_points[canonical_idx] = hybrid_points[canonical_idx]

        temp_canon = utils.CanonObj(tmp_obj_points[canonical_idx], mesh_pcls[part][canonical_idx], mesh_faces[part][canonical_idx], None, None)

        warps, _ = aligned_warp_gen(temp_canon, canonical_idx, tmp_obj_points, obj_contact_points, alpha, visualize=True)
        _, pca = utils.pca_transform(warps, n_dimensions)

        warp_results[part] = {
                "pca": pca,
                "canonical_pcl": hybrid_points[canonical_idx],
                "canonical_mesh_points": mesh_points[canonical_idx],
                "canonical_mesh_faces": faces[canonical_idx], 
                "canonical_center_transform": centers[canonical_idx],
                "canonical_contacts": obj_contact_points[canonical_idx] 
            }
    
    return warp_results



if __name__ == "__main__":
    print("generating")

    num_training = 6
    n_dimensions = 4
    #will need to be made more general later somehow
    obj_type = 'mug'
    part_labels = {'cup': 37, 'handle': 36}
    part_names = ['cup', 'handle']
    directory = "../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/"

    # obj_type = 'syn_rack_easy'
    # part_labels = {'branch': 1, 'trunk': 0}
    # part_names = ['branch', 'trunk']
    # directory = "../relational_ndf/src/rndf_robot/descriptions/objects/syn_racks_easy_obj/"

    obj_type = 'bowl'
    part_labels = {'whole_bowl': 0}
    part_names = ['whole_bowl']
    directory = "../relational_ndf/src/rndf_robot/descriptions/objects/bowl_centered_obj_normalized/"

    print("Sampling")

    #training_ids = [43,98,103,91,8,88,49,100,96,97]
    # all_shapenet_mugs = load_all_shapenet_files(obj_type)
    # filtered_shapenet_mugs = []
    # for candidate_id in all_shapenet_mugs:
    #         try:
    #             load_segmented_pointcloud_from_txt(candidate_id)
    #             filtered_shapenet_mugs.append(candidate_id)
    #         except:
    #             continue


    training_ids = sample_training(num_training, obj_type, directory)
    #np.array(filtered_shapenet_mugs)[np.array([43,98,103,91,8,88,49,100,96,97])]
    training_whole_meshes = []
    training_part_meshes = {part:[] for part in part_names}

    print("Loading objects")
    if obj_type == 'syn_rack_easy':
        for obj_id in training_ids:
            training_whole_meshes.append(get_rack_mesh(obj_id))
            part_meshes = get_segmented_rack_mesh(training_whole_meshes[-1])
            for part in part_names: 
                training_part_meshes[part].append(part_meshes[part_labels[part]])
    elif obj_type == "mug":   
        for obj_id in training_ids:
            training_whole_meshes.append(get_mesh(obj_id))
            part_meshes = get_segmented_mesh(obj_id)
            for part in part_names: 
                training_part_meshes[part].append(part_meshes[part_labels[part]])
    elif obj_type == "bowl":
        for obj_id in training_ids:
            training_whole_meshes.append(get_bowl(obj_id))
        #         part_meshes = get_segmented_rack_mesh(training_whole_meshes[-1])
        #         for part in part_names: 
        #             training_part_meshes[part].append(part_meshes[part_labels[part]])

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #print("Whole objects")
    #training_ids = list(training_ids)

    #Load whole mugs
    #Whole mug warps
    whole_obj_warp_name = f"whole_{obj_type}_{timestr}_{len(training_ids)}"
    whole_obj_tag = 'none'
    whole_obj_warp_data = learn_warps(training_whole_meshes, n_dimensions=n_dimensions)
    print(whole_obj_warp_data['canonical_idx'])
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
    print("Part warping")



    # part_warps = {}
    # for part in part_names:
    #     part_warp_data = learn_warps(training_part_meshes[part], n_dimensions=n_dimensions)

    #     part_warp_name = f"{part}_{timestr}_{len(training_ids)}"
    #     part_tag = 'none'
    #     part_metadata = CanonPartMetadata(part_tag,
    #                                         training_ids[part_warp_data['canonical_idx']],
    #                                         training_ids[:part_warp_data['canonical_idx']] + training_ids[part_warp_data['canonical_idx']+1:],
    #                                         part_label=None)

    #     part_warps[part] = CanonPart(part_warp_data['canonical_pcl'], 
    #                            part_warp_data['canonical_mesh_points'],
    #                            part_warp_data['canonical_mesh_faces'],
    #                             part_warp_data['canonical_center_transform'],                                 
    #                             contact_points=None,
    #                             metadata=part_metadata,
    #                             pca=part_warp_data['pca'],)
    #     pickle.dump(part_warps[part], open(part_warp_name, 'wb'))


    # #do the prealigning
    # for part in args.part_names:
    #     part_ids = training_ids
    #     part_data = load_pcl(training_ids, part_labels[part])

    #     part_warp_name = f"{part}_{timestr}"
    #     part_tag = 'prealign'
    #     part_warps[part]  = CanonPart(part_data['hybrid_pcls'][canonical_idx], 
    #                             part_data['mesh_verts'][canonical_idx],
    #                             part_data['mesh_faces'][canonical_idx],
    #                             part_data['center_transforms'][canonical_idx], part_tag, 
    #                             part_ids[canonical_idx], 
    #                             part_ids[:canonical_idx] + part_ids[canonical_idx+1:], 
    #                             part_labels[part],
    #                             part_pca,
    #                             )

























