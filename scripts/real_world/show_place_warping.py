import argparse
import pickle

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from src import utils


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
    new_mug_pc = utils.transform_pcd(mug_pc, vp_to_p2)
    return new_mug_pc


def main(args):

    if args.task == "mug_tree":
        canon_source = utils.CanonObj.from_pickle("data/230213_ndf_mugs_scale_large_pca_8_dim_alp_0.01.pkl")
        canon_target = utils.CanonObj.from_pickle("data/230213_ndf_trees_scale_large_pca_8_dim_alp_2.pkl")
        source_scale = 0.2
        target_scale = 0.5
    else:
        raise NotImplementedError()

    with open(args.load_path, "rb") as f:
        place_data = pickle.load(f)

    knns = place_data["knns"]
    deltas = place_data["deltas"]
    target_indices = place_data["target_indices"]

    source_pcd = canon_source.to_pcd(utils.ObjParam(latent=[0.] * canon_source.n_components, scale=source_scale))
    smin, smax = -2., 2.
    vmin, vmax = -0.3, 0.3

    print("Showing target closest points:")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    target_pcd = canon_target.to_pcd(utils.ObjParam(latent=[0.] * canon_target.n_components, scale=target_scale))
    show_tree_indices(ax, target_pcd, target_indices, vmin, vmax)
    plt.show()

    print("Showing placement pose warping:")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    update_axis(ax, warp(source_pcd, target_pcd, knns, deltas, target_indices), target_pcd, vmin, vmax)

    slider_axes = []
    z = 0.
    for _ in range(canon_source.n_components):
        slider_axes.append(fig.add_axes([0.25, z, 0.65, 0.03]))
        z += 0.05
    # we start at the bottom and move up
    slider_axes = list(reversed(slider_axes))

    sliders = []
    for i in range(canon_source.n_components):
        sliders.append(Slider(slider_axes[i], "D{:d}".format(i), smin, smax, valinit=0))

    def sliders_on_changed(val):
        latents = np.array([[s.val for s in sliders]])
        source_pcd = canon_source.to_pcd(utils.ObjParam(latent=latents, scale=source_scale))
        update_axis(ax, warp(source_pcd, target_pcd, knns, deltas, target_indices), target_pcd, vmin, vmax)

    for s in sliders:
        s.on_changed(sliders_on_changed)

    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str, help="[mug_tree, bowl_on_mug, bottle_in_box]")
parser.add_argument("load_path")
main(parser.parse_args())
