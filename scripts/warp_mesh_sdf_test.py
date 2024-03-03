import pickle
from scripts.generate_warps import load_all_shapenet_files, get_mesh, get_segmented_mesh, CanonPart, CanonPartMetadata
from scripts.compare_warps_script import whole_object_warping
from mesh_to_sdf import mesh_to_voxels, sample_sdf_near_surface
from src import utils, viz_utils
import numpy as np
import trimesh
import skimage
import pyrender
import mesh2sdf
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform  import Rotation

import torch
from torch import nn, optim


def orthogonalize(x: torch.Tensor) -> torch.Tensor:
    """
    Produce an orthogonal frame from two vectors
    x: [B, 2, 3]
    """
    #u = torch.zeros([x.shape[0],3,3], dtype=torch.float32, device=x.device)
    u0 = x[:, 0] / torch.norm(x[:, 0], dim=1)[:, None]
    u1 = x[:, 1] - (torch.sum(u0 * x[:, 1], dim=1))[:, None] * u0
    u1 = u1 / torch.norm(u1, dim=1)[:, None]
    u2 = torch.cross(u0, u1, dim=1)
    return torch.stack([u0, u1, u2], dim=1)



def sdf_from_mesh(mesh):
    pass

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sample_transformed_sdf(X_h, sdf, center_param, pose_param):
    rotm = orthogonalize(pose_param)
    transformed_X_h = torch.bmm(X_h, rotm.permute((0, 2, 1))) + center_param[:, None]
    return sdf(transformed_X_h)

def collision_functional(sdf_1_samples, sdf_2_samples, alpha = 1 ):
    return sum(sigmoid(-alpha*sdf_1_samples) * sigmoid(-alpha*sdf_2_samples))

def grid_sample(bbox_mins, bbox_maxes, num_axis_samples):
    nx, ny, nz = (num_axis_samples, num_axis_samples, num_axis_samples)
    x = np.linspace(bbox_mins[0], bbox_maxes[0], nx)
    y = np.linspace(bbox_mins[1], bbox_maxes[1], ny)
    z = np.linspace(bbox_mins[2], bbox_maxes[2], nz)
    xv, yv, zv = np.meshgrid(x, y, z)
    X_h = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T
    return X_h

#load warp
#load test object
#display canonical mesh
if __name__ == "__main__":
    #load warp
    #load demo object
    #get the demo object pose
    #do the same with the target object and pose
    #use that to get a bounding box of space and initial guess?
    #or we could just start by giving each of them the same center and orientation and perturbing it
    warp_file_stamp = '20240202-160637'

    #todo: generalize for other objects
    object_warp_file = f'whole_mug_{warp_file_stamp}'
    cup_warp_file = f'cup_{warp_file_stamp}'
    handle_warp_file = f'handle_{warp_file_stamp}'

    parent_warp_file = "data/pcas/230315_ndf_trees_scale_pca_8_dim_alp_0_01.pkl"

    part_names = ['cup', 'handle']
    part_labels = {'cup': 37, 'handle':36}

    part_canonicals = {}
    whole_child_canonical = pickle.load(open( object_warp_file, 'rb'))
    whole_parent_canonical = utils.CanonObj.from_pickle(parent_warp_file)
    part_canonicals['cup'] = pickle.load(open( cup_warp_file, 'rb'))
    part_canonicals['handle'] = pickle.load(open( handle_warp_file, 'rb'))

    # whole_mesh = trimesh.Trimesh(vertices=whole_object_canonical.mesh_vertices, faces=whole_object_canonical.mesh_faces)
   

    target_id='5c7c4cb503a757147dbda56eabff0c47'
    target_whole_mesh = get_mesh(target_id)
    target_part_meshes = get_segmented_mesh(target_id)
    rotation = Rotation.from_euler("yxz", [np.pi * np.random.random(), np.pi * np.random.random(), np.pi * np.random.random()]).as_matrix()
    utils.trimesh_transform(target_whole_mesh, center=False, rotation=rotation)

    for part in part_names:
        utils.trimesh_transform(target_part_meshes[part_labels[part]], center=False, rotation=rotation)

    #whole_mesh.show()
    #target_whole_mesh.show()

    # target_whole = utils.trimesh_create_verts_surface(target_whole_mesh, num_surface_samples=2000)
    # target_parts = {part:utils.trimesh_create_verts_surface(target_part_meshes[part_labels[part]], num_surface_samples=2000) for part in part_names}

    #whole_warped, whole_warped_mesh = whole_object_warping(target_whole, whole_object_canonical)


    from pysdf import SDF
    whole_canon_child_sdf = SDF(whole_child_canonical.mesh_vertices, whole_child_canonical.mesh_faces)
    whole_canon_parent_sdf = SDF(whole_parent_canonical.mesh_vertices, whole_parent_canonical.mesh_faces)

    #get the bounding box as the max bb of both meshes plus some amount? 
    #load both and just visualize them in a scene together for the sake of clarity
    #

    viz_utils.show_pcds_plotly({ 'child': whole_child_canonical.canonical_pcl, 'parent': whole_parent_canonical.canonical_pcl})

    bb_max = np.max(np.concatenate([whole_child_canonical.mesh_vertices, whole_parent_canonical.mesh_vertices,], axis=0), axis=0)
    bb_min = np.min(np.concatenate([whole_child_canonical.mesh_vertices, whole_parent_canonical.mesh_vertices,], axis=0), axis=0)

    print(bb_min, bb_max)
    optimizer_steps = 1000
    #whole_warped_sdf = SDF(whole_warped_mesh.vertices, whole_warped_mesh.faces)

    #sample a bunch of relative transforms 
    X_h = torch.from_numpy(grid_sample(bb_min, bb_max, 64))

    n_angles = 12
    #setup optimizer
    unit_ortho = np.array([
            [1., 0., 0.],
            [0., 1., 0.]
    ], dtype=np.float32)
    unit_ortho = np.repeat(unit_ortho[None], n_angles, axis=0)
    init_ortho_pt = torch.tensor(unit_ortho, dtype=torch.float32, device="cpu")
    initial_centers_pt = torch.zeros((n_angles, 3), dtype=torch.float32, device="cpu")
    
    center_param = nn.Parameter(initial_centers_pt, requires_grad=True)
    pose_param = nn.Parameter(init_ortho_pt, requires_grad=True)

    params = [center_param, pose_param]
    optim = optim.Adam(params, lr=.01)

    for step in range(optimizer_steps):
        center = center_param
        orn = pose_param
        optim.zero_grad()
        child_samples = sample_transformed_sdf(X_h, whole_canon_child_sdf, center_param, pose_param, )
        parent_samples = whole_canon_parent_sdf(X_h)

        cost = collision_functional(child_samples, parent_samples)
        cost.sum().backward()
        self.optim.step()

    with torch.no_grad:
        viz_utils.show_pcds_plotly({ 'child': utils.transform_pcd(whole_child_canonical.canonical_pcl, utils.pos_quat_to_transform(center_param, utils.rotm_to_quat(pose_param)))
            , 'parent': whole_parent_canonical.canonical_pcl})

    # mesh_scale = 0.8
    # size = 128
    # level = 2 / size
    # levels = [-0.02, 0.0, 0.02]

    # # normalize mesh
    # vertices = whole_warped_mesh.vertices
    # bbmin = vertices.min(0)
    # bbmax = vertices.max(0)
    # center = (bbmin + bbmax) * 0.5
    # scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    # vertices = (vertices - center) * scale

    # # fix mesh
    # t0 = time.time()
    # sdf, mesh = mesh2sdf.compute(
    #     vertices, whole_warped_mesh.faces, size, fix=True, level=level, return_mesh=True)
    # print(type(sdf))
    # print(sdf.shape)
    # exit(0)
    # t1 = time.time()

    # # output
    # # mesh.vertices = mesh.vertices / scale + center
    # # mesh.export(filename[:-4] + '.fixed.obj')
    # # np.save(filename[:-4] + '.npy', sdf)
    # print('It takes %.4f seconds to process' % (t1-t0))

    # # extract level sets
    # for i, level in enumerate(levels):
    #   vtx, faces, _, _ = skimage.measure.marching_cubes(sdf, level)

    #   vtx = vtx * (mesh_scale * 2.0 / size) - 1.0
    #   # mesh = trimesh.Trimesh(vtx, faces)
    #   # mesh.show()
    #   #mesh.export(os.path.join(folder, 'l%.2f.obj' % level))


    # # draw image
    # for i in range(size):
    #   array_2d = sdf[:, :, i]

    #   num_levels = 6
    #   fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)
    #   levels_pos = np.logspace(-2, 0, num=num_levels)  # logspace
    #   levels_neg = -1. * levels_pos[::-1]
    #   levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
    #   colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))

    #   sample = array_2d
    #   # sample = np.flipud(array_2d)
    #   CS = ax.contourf(sample, levels=levels, colors=colors)

    #   ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    #   ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    #   ax.axis('off')
    #   plt.show()

    # # points, sdf = sample_sdf_near_surface(whole_mesh, number_of_points=250000)

    # # colors = np.zeros(points.shape)
    # # colors[sdf < 0, 2] = 1
    # # #colors[sdf > 0, 0] = 1
    # # cloud = pyrender.Mesh.from_points(points[colors[:, 2] != 0], colors=colors[colors[:, 2] != 0])
    # # scene = pyrender.Scene()
    # # scene.add(cloud)
    # # viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

    # # utils.trimesh_transform(whole_mesh, center=False, scale=10)
    # # voxels = mesh_to_voxels(whole_mesh, 64, pad=True,)

    # # vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
    # # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    # # mesh.show()


