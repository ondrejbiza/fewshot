import argparse
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pickle

import utils


def show_tree_indices(ax, tree_pc: NDArray, indices: NDArray[np.int32], vmin: float, vmax: float):

    ax.clear()
    ax.scatter(tree_pc[:, 0], tree_pc[:, 1], tree_pc[:, 2], color="red", alpha=0.5)
    ax.scatter(tree_pc[indices, 0], tree_pc[indices, 1], tree_pc[indices, 2], color="green", s=50)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_zlim(vmin, vmax)


def update_axis(ax, new_obj: NDArray, tree: NDArray, vmin: float, vmax: float):

    ax.clear()
    ax.scatter(new_obj[:, 0], new_obj[:, 1], new_obj[:, 2], color="red", alpha=0.5)
    ax.scatter(tree[:, 0], tree[:, 1], tree[:, 2], color="green", alpha=0.5)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_zlim(vmin, vmax)


def warp(mug_pc, tree_pc, knns, deltas, target_indices):

    anchors = mug_pc[knns]
    targets = np.mean(anchors + deltas, axis=1)

    points_2 = tree_pc[target_indices]

    vp_to_p2, _, _ = utils.best_fit_transform(targets, points_2)
    new_mug_pc = utils.transform_pointcloud_2(mug_pc, vp_to_p2)
    return new_mug_pc


def main(args):

    with open("data/ndf_mugs_pca_4_dim.npy", "rb") as f:
        canon_mug = pickle.load(f)

    with open("data/real_tree_pc.pkl", "rb") as f:
        canon_tree = pickle.load(f)

    with open("data/real_place_clone.pkl", "rb") as f:
        place_data = pickle.load(f)

    knns = place_data["knns"]
    deltas = place_data["deltas"]
    target_indices = place_data["target_indices"]

    new_obj = utils.canon_to_pc(canon_mug, [np.array([[0.] * canon_mug["pca"].n_components])])
    smin, smax = -2., 2.
    vmin, vmax = -0.3, 0.3

    print("Showing tree closest points:")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    show_tree_indices(ax, canon_tree["canonical_obj"], target_indices, vmin, vmax)
    plt.show()

    print("Showing placement pose warping:")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    update_axis(ax, warp(new_obj, canon_tree["canonical_obj"], knns, deltas, target_indices), canon_tree["canonical_obj"], vmin, vmax)

    slider_axes = []
    z = 0.
    for _ in range(canon_mug["pca"].n_components):
        slider_axes.append(fig.add_axes([0.25, z, 0.65, 0.03]))
        z += 0.05
    # we start at the bottom and move up
    slider_axes = list(reversed(slider_axes))

    sliders = []
    for i in range(canon_mug["pca"].n_components):
        sliders.append(Slider(slider_axes[i], "D{:d}".format(i), smin, smax, valinit=0))

    def sliders_on_changed(val):
        latents = np.array([[s.val for s in sliders]])
        new_obj = utils.canon_to_pc(canon_mug, [latents])
        update_axis(ax, warp(new_obj, canon_tree["canonical_obj"], knns, deltas, target_indices), canon_tree["canonical_obj"], vmin, vmax)

    for s in sliders:
        s.on_changed(sliders_on_changed)

    plt.show()


parser = argparse.ArgumentParser()
main(parser.parse_args())
