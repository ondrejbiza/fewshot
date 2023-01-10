from functools import partial
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import torch
import trimesh
import trimesh.voxel.creation as vcreate
from trimesh import decomposition
from trimesh.scene import scene
from pycpd import DeformableRegistration
from sklearn.decomposition import PCA


def load_object(obj_path):

    return trimesh.load(obj_path)


def load_object_create_verts(obj_path, voxel_size=0.015, center=True, scale=None, rotation=None):
    # loads a mesh (STL), samples  points from its watertight mesh using voxelization
    # !!! centers the resulting point cloud, but doesn't scale it
    mesh = trimesh.load(obj_path)

    if scale is not None or rotation is not None:
        # Apply transforms.
        transform = np.eye(3)
        if scale is not None:
            transform[0, 0] *= scale
            transform[1, 1] *= scale
            transform[2, 2] *= scale
        if rotation is not None:
            transform = np.matmul(rotation, transform)

        tmp = np.eye(4)
        tmp[:3, :3] = transform
        mesh.apply_transform(tmp)

    voxels = vcreate.voxelize(mesh, voxel_size)
    verts = np.array(voxels.points, dtype=np.float32)
    print("Num. vertices: {:d}".format(len(verts)))

    if center:
        print(np.mean(verts, axis=0))
        verts -= np.mean(verts, axis=0)

    return verts


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

    overall_costs = []
    for i in range(len(known_pts)):
        print(i)
        cost_per_target = []
        for j in range(len(known_pts)):
            if i != j:
                cost_per_target.append(cost_batch(known_pts[i], known_pts[j]))
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
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.1, 0.1)
    plt.draw()
    plt.pause(0.001)


def cpd_transform_plot(source, target):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)
    reg = DeformableRegistration(**{ 'X': source, 'Y': target, 'tolerance':0.00001 })
    reg.register(callback)
    #Returns the gaussian means and their weights - WG is the warp of source to target
    return reg.W, reg.G


def cpd_transform(source, target):
    reg = DeformableRegistration(**{ 'X': source, 'Y': target, 'tolerance':0.00001 })
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

        if visualize:
            w, g = cpd_transform_plot(target, source)
        else:
            w, g = cpd_transform(target, source)

        plt.clf()
        plt.close()

        warp = np.dot(g, w)
        warps.append(np.hstack(warp))

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
