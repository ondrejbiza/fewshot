from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import trimesh.voxel.creation as vcreate
from pycpd import DeformableRegistration
from sklearn.decomposition import PCA

""" Coherent Point Drift """


def visualize(iteration, error, X, Y, ax):
    # Displays the current transform
    # Uncomment below to display the transforms
    # plt.cla()
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color="red", label="Source")
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color="blue", label="Target")
    # ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    # ax.legend(loc='upper left', fontsize='x-large')
    # plt.draw()
    # plt.pause(0.001)


# Warps the source point cloud to the target pointcloud
def cpd_transform(source, target):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    callback = partial(visualize, ax=ax)
    reg = DeformableRegistration(**{"X": source, "Y": target, "tolerance": 0.00001})
    reg.register(callback)
    # Returns the gaussian means and their weights - WG is the warp of source to target
    return reg.W, reg.G


def generate_warps(source, targets, num=0):
    warps = []
    i = 0
    for target in targets:
        i += 1
        w, g = cpd_transform(target, source)
        warp = np.dot(g, w)
        warps.append(np.hstack(warp))
    np.save("cup_warps_" + str(num), warps)
    return warps


""" Picking canonical objects by comparing object pointclouds """


def cost(source, target):
    total_cost = 0
    i = 0
    for point in source:
        idx = (np.sum(np.abs(point - target), axis=1)).argmin()
        total_cost += np.linalg.norm(point - target[idx])
    return total_cost


def pick_canonical(known_pts):
    overall_costs = []
    for i in range(len(known_pts)):
        cost_per_target = []
        for j in range(len(known_pts)):
            if i != j:
                cost_per_target.append(cost(known_pts[i], known_pts[j]))
        overall_costs.append(np.mean(cost_per_target))
    return np.argmin(overall_costs)


""" PCA Latent Space and reconstruction """


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

    def cost(recreated_target, goal):
        total_cost = 0
        i = 0
        for point in recreated_target:

            idx = (np.sum(np.abs(point - goal), axis=1)).argmin()
            total_cost += np.linalg.norm(point - goal[idx])

        return total_cost

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
    print(np.min(costs))
    return samples[np.argmin(costs)]


""" Loading objects """

# For loading objects from the obj files
def load_obj(filename, val=0.45):
    from pywavefront import Wavefront

    scene = Wavefront(filename, parse=False)
    scene.parse()  # Explicit call to parse() needed when parse=False
    verts = None
    for name, material in scene.materials.items():
        # Contains the vertex list of floats in the format described above
        mesh = trimesh.load(filename)
        voxels = vcreate.voxelize(
            mesh, val
        )  # Sampling density. 5 for cups, .05 for spatulas
        verts = voxels.points

        # hack - this file is rotated from the others for some reason
        if (
            filename == "./cup_models/glass_06_cd.obj"
            or filename == "./cup_models/glass_06.obj"
        ):
            a = verts.T[np.ix_([2, 0, 1])]
            verts = a.T
    return verts


def warp_gen(canonical_index, objects):
    scale_factor = 0.2  # CPD is sensitive to scale, got this through trial and error
    source = np.array(objects[canonical_index])  # * scale_factor
    # print(source.shape)
    targets = np.array(objects[1:])  # * scale_factor
    warps = []
    for target in targets:
        # print(target.shape)
        w, g = cpd_transform(target, source)
        warp = np.dot(g, w)
        warps.append(np.hstack(warp))

        plt.clf()
        plt.close()
    return warps


if __name__ == "__main__":
    pass
    # # Example use case
    # spatula_names = ["spatula_1.obj", "spatula_2.obj", "spatula_3.obj", "spatula_4.obj", "spatula_5.obj", "spatula_6.obj"]
    # test_spatula = load_obj("spatula_7.obj") * .2 #scale factor hack, see warp gen function

    # pointclouds = []
    # for spatula in spatula_names:
    #     pointclouds.append(load_obj(spatula))
    # canonical_index = pick_canonical(pointclouds)

    # warps = warp_gen(canonical_index, pointclouds)
    # components, pca = pca_transform(warps)
    # latent_rep = pca_recconstruct(components, pca, pointclouds[canonical_index], test_spatula)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # source = pointclouds[canonical_index]
    # reconst = source + pca.inverse_transform(np.atleast_2d(latent_rep)).inverse_transform(sample).reshape((1000, 3))
    # target = test_spatula
    # ax.scatter(source[:,0],  source[:,1], source[:,2], color='red', label='Source')
    # ax.scatter(target[:,0],  target[:,1], target[:,2], color='blue', label='Target')
    # ax.scatter(reconst[:,0],  reconst[:,1], reconst[:,2], color='purple', label='Reconst')

    # plt.show()
