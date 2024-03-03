from src.demo import get_closest_point_pairs, fit_plane_points, get_closest_point_pairs_thresh
from src import utils, viz_utils
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

#Returns the unit vector for vector
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

#Returns the angle between vectors
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def load_segmented_pointcloud_from_txt(pcl_id, num_points=2048, root = 'Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390'):
    fn = os.path.join(root, pcl_id + '.txt')
    cls = 'Mug'
    data = np.loadtxt(fn).astype(np.float32)
    #if not self.normal_channel:
    point_set = data[:, 0:3]
    #else:
        #point_set = data[:, 0:6]
    seg_ids = data[:, -1].astype(np.int32)

    point_set[:, 0:3] = center_pcl(point_set[:, 0:3])

    rotation = Rotation.from_euler("zyx", [0., np.pi/2, 0.]).as_quat()

    transform = utils.pos_quat_to_transform([0,0,0], rotation)
    point_set = utils.transform_pcd(point_set, transform)

    choice = np.random.choice(len(seg_ids), num_points, replace=True)
    # resample
    #point_set = point_set[choice, :]
    #seg = seg[choice]

    return point_set, cls, seg_ids

def center_pcl(pcl, return_centroid=False):
    l = pcl.shape[0]
    centroid = np.mean(pcl, axis=0)
    pcl = pcl - centroid
    if return_centroid:
        return pcl, centroid
    else:
        return pcl

#load the demo and canonical cup
source_part_names = ['cup', 'handle']
demo_pcl_id = '5c7c4cb503a757147dbda56eabff0c47'
canon_source_path = 'data/1234_part_based_mugs_4_dim.pkl'


demo_mug, _, demo_seg_ids = load_segmented_pointcloud_from_txt(demo_pcl_id)

rotation = Rotation.from_euler("xyz", [np.pi/2., 0., np.pi/12.]).as_quat()

demo_mug = utils.transform_pcd(demo_mug, utils.pos_quat_to_transform([0,0,0], rotation))

demo_cup = demo_mug[demo_seg_ids==37]
demo_handle = demo_mug[demo_seg_ids==36]

cup_path = './scripts/cup_parts/m1.obj'
cup_handle_path = './scripts/handles/m1.obj'

handle_path = './scripts/handles/m4.obj'
handle_cup_path  = './scripts/cup_parts/m4.obj'

def load_obj_part(obj_path, return_faces = False):
    mesh = utils.trimesh_load_object(obj_path)
    rotation = Rotation.from_euler("zyx", [0., 0., np.pi/2]).as_matrix()
    utils.trimesh_transform(mesh, center=False, scale=None, rotation=rotation)
    ssp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
    if return_faces: 
        mp, f = utils.trimesh_get_vertices_and_faces(mesh)
        return ssp, mp, f
    return ssp


canon_source_parts = utils.CanonObj.from_parts_pickle(canon_source_path, source_part_names)
#canon_source = utils.CanonObj.unified_from_parts_pickle(canon_source_path, source_part_names)


canon_cup, canon_cup_handle = load_obj_part(cup_path), load_obj_part(cup_handle_path)
canon_handle, canon_handle_cup = load_obj_part(handle_path), load_obj_part(handle_cup_path)

canon_cup, canon_cup_handle, canon_handle, canon_handle_cup = utils.scale_points_circle([canon_cup, canon_cup_handle, canon_handle, canon_handle_cup], base_scale=0.125)
demo_cup, demo_handle, demo_mug = utils.scale_points_circle([demo_cup, demo_handle, demo_mug], base_scale=0.125)
canon_cup_mug = np.concatenate([canon_cup, canon_cup_handle])
canon_handle_mug = np.concatenate([canon_handle, canon_handle_cup])

canon_handle, handle_mp, handle_faces = load_obj_part(handle_path, return_faces=True)
canon_handle_cup, handle_cup_mp, handle_cup_faces = load_obj_part(handle_cup_path, return_faces=True)
handle_mug_vertices = np.concatenate([handle_mp, handle_cup_mp])
handle_mug_faces = np.concatenate([handle_faces, handle_cup_faces + len(handle_mp)])

canon_handle_mug_canon_wrapper = utils.CanonObj(canon_handle_mug, handle_mug_vertices, handle_mug_faces, utils.pos_quat_to_transform(np.mean(canon_handle_mug, 0), [0,0,0,1])) 


#do the aligning to get the mug alignments and check results? 
from src.object_warping import ObjectSE2Batch, WarpBatch, warp_to_pcd, align_to_pcd_se2, PARAM_1
import copy as cp

inference_kwargs = {
            "train_latents": False,
            "train_scales": True,
            "train_poses": True,
        }

aligned_mug = ObjectSE2Batch(
                canon_handle_mug_canon_wrapper, demo_mug, 'cpu', **cp.deepcopy(PARAM_1),
                init_scale=1) 
aligned_handle_mug, _, aligned_handle_mug_params = align_to_pcd_se2(aligned_mug, n_angles=18, n_batches=1, inference_kwargs=inference_kwargs)

ax = plt.subplot(111, projection='3d')
ax.set_xlim(-.25, .25)
ax.set_ylim(-.25, .25)
ax.set_zlim(-.25, .25)
ax.scatter(canon_handle_mug[:, 0], canon_handle_mug[:, 1], canon_handle_mug[:, 2], color='g')
ax.scatter(aligned_handle_mug[:, 0], aligned_handle_mug[:, 1], aligned_handle_mug[:, 2], color='b', alpha=0.05)
#ax.quiver(0, 0, -.25, canon_handle_normal[0], canon_handle_normal[1], canon_handle_normal[2], color="b")
#ax.scatter(source[:, 0], source[:, 1], source[:, 2], color='g')
#ax.quiver(0, 0, -.25, demo_normal[0], demo_normal[1], demo_normal[2], color="g")
#ax.scatter(canon_aligned_demo_handle[:, 0], canon_aligned_demo_handle[:, 1], canon_aligned_demo_handle[:, 2], color='r')
ax.scatter(demo_mug[:, 0], demo_mug[:, 1], demo_mug[:, 2], color='r')
plt.show()

