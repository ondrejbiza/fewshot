import copy as cp
from typing import Any, Optional, Tuple, Dict
from functools import partial
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import torch
import trimesh
import trimesh.voxel.creation as vcreate
from trimesh import decomposition
from trimesh.scene import scene
#from pycpd import DeformableRegistration
from cycpd import deformable_registration
from sklearn.decomposition import PCA
from src import viz_utils


def load_object(obj_path):

    return trimesh.load(obj_path)


def load_object_create_verts(
    obj_path: str, voxel_size: float=0.015, center: bool=True,
    scale: Optional[float]=None, rotation: Optional[NDArray]=None,
    sampling_method: str="hybrid", num_surface_samples: Optional[int]=1500
    ) -> Dict[str, Optional[NDArray]]:

    mesh = trimesh.load(obj_path, force=True)

    # Automatically center. Also possibly rotate and scale.
    translation_matrix = np.eye(4)
    scaling_matrix = np.eye(4)
    rotation_matrix = np.eye(4)

    if center:
        t = mesh.centroid
        translation_matrix[:3, 3] = -t

    if scale is not None:
        scaling_matrix[0, 0] *= scale
        scaling_matrix[1, 1] *= scale
        scaling_matrix[2, 2] *= scale

    if rotation is not None:
        rotation_matrix[:3, :3] = rotation

    transform = np.matmul(scaling_matrix, np.matmul(rotation_matrix, translation_matrix))
    mesh.apply_transform(transform)

    out: Dict[str, Optional[NDArray]] = {
        "mesh_points": None,
        "surface_points": None,
        "volume_points": None,
        "faces": None,
        "points": None,
    }

    if sampling_method == "volume":
        voxels = vcreate.voxelize(mesh, voxel_size)
        points = np.array(voxels.points, dtype=np.float32)

        out["volume_points"] = points
        out["points"] = points
    elif sampling_method == "hybrid":
        surf_points, _ = trimesh.sample.sample_surface_even(
            mesh, num_surface_samples
        )
        mesh_points = np.array(mesh.vertices)
        # Crucial that the mesh points come first!
        points = np.concatenate([mesh_points, surf_points])

        out["surface_points"] = surf_points
        out["mesh_points"] = mesh_points
        out["faces"] = mesh.faces
        out["points"] = points
    elif sampling_method == "surface":
        surf_points, _ = trimesh.sample.sample_surface_even(
            mesh, num_surface_samples
        )

        out["surface_points"] = surf_points
        out["points"] = surf_points
    else:
        raise ValueError("Invalid point cloud sampling method.")

    print("Num. points:")
    for k, v in out.items():
        if v is not None:
            print(k, len(v))

    return out


def cost(source: NDArray, target: NDArray) -> float:

    total_cost = 0
    for point in source:
        idx = (np.sum(np.abs(point - target), axis=1)).argmin()
        total_cost += np.linalg.norm(point - target[idx])
    return total_cost / len(source)  # TODO: test averaging instead of sum


def cost_batch(source: NDArray, target: NDArray) -> float:

    idx = np.sum(np.abs(source[None, :] - target[:, None]), axis=2).argmin(axis=0)
    return np.mean(np.linalg.norm(source - target[idx], axis=1))  # TODO: test averaging instead of sum


def cost_pt(source, target):

    source_d, target_d = source.detach(), target.detach()
    indices = (source_d[:, None] - target_d[None, :]).square().sum(dim=2).argmin(dim=1)

    c = torch.sqrt(torch.sum(torch.square(source - target[indices]), dim=1))
    return torch.mean(c)

    # diff = torch.sqrt(torch.sum(torch.square(source[:, None] - target[None, :]), dim=2))
    # c = torch.min(diff, dim=1)[0]
    # return torch.mean(c)


def cost_batch_pt(source, target):

    # for each vertex in source, find the closest vertex in target
    # we don't need to propagate the gradient here
    source_d, target_d = source.detach(), target.detach()
    indices = (source_d[:, :, None] - target_d[:, None, :]).square().sum(dim=3).argmin(dim=2)

    # go from [B, indices_in_target, 3] to [B, indices_in_source, 3] using target[batch_indices, indices]
    batch_indices = torch.arange(0, indices.size(0), device=indices.device)[:, None].repeat(1, indices.size(1))
    c = torch.sqrt(torch.sum(torch.square(source - target[batch_indices, indices]), dim=2))
    return torch.mean(c, dim=1)

    # simple version, about 2x slower
    # bigtensor = source[:, :, None] - target[:, None, :]
    # diff = torch.sqrt(torch.sum(torch.square(bigtensor), dim=3))
    # c = torch.min(diff, dim=2)[0]
    # return torch.mean(c, dim=1)


def pick_canonical(known_pts):

    # GPU acceleration makes this at least 100 times faster.
    known_pts = [torch.tensor(x, device="cuda:0", dtype=torch.float32) for x in known_pts]

    overall_costs = []
    for i in range(len(known_pts)):
        print(i)
        cost_per_target = []
        for j in range(len(known_pts)):
            if i != j:
                with torch.no_grad():
                    cost = cost_batch_pt(known_pts[i][None], known_pts[j][None]).cpu()

                cost_per_target.append(cost.item())

        overall_costs.append(np.mean(cost_per_target))
    print("overall costs: {:s}".format(str(overall_costs)))
    return np.argmin(overall_costs)


def visualize(iteration, error, X, Y, ax):
    #Displays the current transform
    #Uncomment below to display the transforms
    plt.cla()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Source')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Target')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    plt.draw()
    plt.pause(0.001)


def cpd_transform_plot(source, target):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)
    # reg = DeformableRegistration(**{ 'X': source, 'Y': target, 'tolerance':0.00001 })
    reg = deformable_registration(**{ 'X': source, 'Y': target, 'tolerance':0.00001 })
    reg.register(callback)
    #Returns the gaussian means and their weights - WG is the warp of source to target
    return reg.W, reg.G


def cpd_transform(source, target):
    # reg = DeformableRegistration(**{ 'X': source, 'Y': target, 'tolerance':0.00001 })
    reg = deformable_registration(**{ 'X': source, 'Y': target, 'tolerance':0.00001 })
    reg.register()
    #Returns the gaussian means and their weights - WG is the warp of source to target
    return reg.W, reg.G


def warp_gen(canonical_index, objects, scale_factor=1., visualize=False):

    source = objects[canonical_index] * scale_factor
    targets = []
    for obj_idx, obj in enumerate(objects):
        if obj_idx != canonical_index:
            targets.append(obj * scale_factor)

    warps = []
    for target_idx, target in enumerate(targets):
        print("target {:d}".format(target_idx))

        # if visualize:
        #     w, g = cpd_transform_plot(target, source)
        # else:
        w, g = cpd_transform(target, source)

        plt.clf()
        plt.close()

        warp = np.dot(g, w)
        warp = np.hstack(warp)
        warps.append(warp)

        if visualize:
            viz_utils.show_pcds_plotly({
                "target": target,
                "warp": source + warp.reshape(-1, 3),
            }, center=True)

    return warps


def pca_transform(distances, n_dimensions=4):

    pca = PCA(n_components=n_dimensions)
    p_components = pca.fit_transform(np.array(distances))
    return p_components, pca


def pca_reconstruct(p_components, pca, source, target, n_samples=300):
    # Sample a bunch of parameters from the PCA space - by taking the known values \
    # and uniformly sampling from their range
    maximums = np.max(p_components, axis=0)
    minimums = np.min(p_components, axis=0)
    samples = []

    for i in range(n_samples):
        sample = np.random.uniform(minimums, maximums)
        samples.append(sample)

        # To Visualize:
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # X = source + PCA.inverse_transform(sample).reshape((1000, 3))
        # ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='RTarget')
        # plt.show()
    costs = []
    for sample in samples:
        warp = pca.inverse_transform(np.atleast_2d(sample))
        tgt = source + np.reshape(warp, source.shape)
        costs.append(cost(target, tgt))

    return samples[np.argmin(costs)]


def convex_decomposition(load_path, save_path):

    mesh = load_object(load_path)
    convex_meshes = decomposition.convex_decomposition(
        mesh, resolution=1000000, depth=20, concavity=0.0025, planeDownsampling=4, convexhullDownsampling=4,
        alpha=0.05, beta=0.05, gamma=0.00125, pca=0, mode=0, maxNumVerticesPerCH=256, minVolumePerCH=0.0001,
        convexhullApproximation=1, oclDeviceID=0
    )
    decomposed_scene = scene.Scene()

    for i, convex_mesh in enumerate(convex_meshes):
        decomposed_scene.add_geometry(convex_mesh, node_name="hull_{:d}".format(i))

    decomposed_scene.export(save_path, file_type="obj")


def scale_object_circle(obj: Dict[str, Any], base_scale: float=1.) -> Dict[str, Any]:
    obj = cp.deepcopy(obj)
    pcd = obj["points"]

    assert len(pcd.shape) == 2
    length = np.sqrt(np.sum(np.square(pcd), axis=1))
    max_length = np.max(length, axis=0)

    for key in ["surface_points", "volume_points", "points"]:
        if obj[key] is not None:
            obj[key] = base_scale * (obj[key] / max_length)

    return obj
