import argparse
import os
import pickle

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from src import utils, viz_utils
from src.real_world import constants


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
    return new_mug_pc, vp_to_p2


def main(args):

    if args.task == "mug_tree":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_MUGS_PCA_PATH)
        canon_target = utils.CanonObj.from_pickle(constants.SIMPLE_TREES_PCA_PATH)
        source_scale = constants.NDF_MUGS_INIT_SCALE
        target_scale = constants.SIMPLE_TREES_INIT_SCALE
    elif args.task == "bowl_on_mug":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_BOWLS_PCA_PATH)
        canon_target = utils.CanonObj.from_pickle(constants.NDF_MUGS_PCA_PATH)
        source_scale = constants.NDF_BOWLS_INIT_SCALE
        target_scale = constants.NDF_MUGS_INIT_SCALE
    elif args.task == "bottle_in_box":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_BOTTLES_PCA_PATH)
        canon_target = utils.CanonObj.from_pickle(constants.BOXES_PCA_PATH)
        source_scale = constants.NDF_BOTTLES_INIT_SCALE
        target_scale = constants.BOXES_INIT_SCALE
    else:
        raise NotImplementedError()

    with open(args.load_path, "rb") as f:
        place_data = pickle.load(f)

    knns = place_data["knns"]
    deltas = place_data["deltas"]
    target_indices = place_data["target_indices"]
    print("# target_indices:", len(target_indices))
    source_param = place_data["source_param"]
    target_param = place_data["target_param"]

    source_pcd = canon_source.to_pcd(utils.ObjParam(latent=source_param.latent, scale=source_param.scale))
    smin, smax = -2., 2.
    vmin, vmax = -0.3, 0.3

    # print("Showing target closest points:")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    target_param = utils.ObjParam(latent=target_param.latent, scale=target_param.scale)
    target_pcd = canon_target.to_pcd(target_param)
    target_mesh = canon_target.to_mesh(target_param)
    # show_tree_indices(ax, target_pcd, target_indices, vmin, vmax)
    # plt.show()

    print("Showing placement pose warping:")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    update_axis(ax, warp(source_pcd, target_pcd, knns, deltas, target_indices)[0], target_pcd, vmin, vmax)

    slider_axes = []
    z = 0.
    for _ in range(canon_source.n_components + 3 + 1):
        slider_axes.append(fig.add_axes([0.25, z, 0.65, 0.03]))
        z += 0.03
    # we start at the bottom and move up
    slider_axes = list(reversed(slider_axes))

    sliders = []
    scale_sliders = []
    for i in range(canon_source.n_components):
        sliders.append(Slider(slider_axes[i], "D{:d}".format(i), smin, smax, valinit=source_param.latent[i]))
    for i in range(3):
        scale_sliders.append(Slider(slider_axes[canon_source.n_components + i], "S{:d}".format(i), smin, smax, valinit=source_param.scale[i]))
    button = Button(slider_axes[canon_source.n_components + 3], "Save pcd")

    def sliders_on_changed(val):
        latents = np.array([[s.val for s in sliders]])
        scale = np.array([s.val for s in scale_sliders])

        source_pcd = canon_source.to_pcd(utils.ObjParam(latent=latents, scale=scale))
        update_axis(ax, warp(source_pcd, target_pcd, knns, deltas, target_indices)[0], target_pcd, vmin, vmax)

    def button_on_changed(val):
        latents = np.array([[s.val for s in sliders]])
        scale = np.array([s.val for s in scale_sliders])

        tmp_source_param = utils.ObjParam(latent=latents, scale=scale)
        source_pcd = canon_source.to_pcd(tmp_source_param)

        anchors = source_pcd[knns]
        targets = np.mean(anchors + deltas, axis=1)

        points_2 = target_pcd[target_indices]

        trans, _, _ = utils.best_fit_transform(targets, points_2)
        source_pcd = utils.transform_pcd(source_pcd, trans)
        targets_trans = utils.transform_pcd(targets, trans)

        pos, quat = utils.transform_to_pos_quat(trans)
        tmp_source_param.position = pos
        tmp_source_param.quat = quat
        source_mesh = canon_source.to_transformed_mesh(tmp_source_param)

        dir_path = "data/warping_figure_a_2"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        i = 1
        for i in range(1, 1000):
            file_path = os.path.join(dir_path, f"source_{i}.pcd")
            if not os.path.isfile(file_path):
                break

        source_mesh.export(os.path.join(dir_path, f"source_{i}.stl"))
        target_mesh.export(os.path.join(dir_path, f"target_{i}.stl"))
        viz_utils.save_o3d_pcd(warp(source_pcd, target_pcd, knns, deltas, target_indices)[0], os.path.join(dir_path, f"source_{i}.pcd"))
        viz_utils.save_o3d_pcd(target_pcd, os.path.join(dir_path, f"target_{i}.pcd"))
        viz_utils.save_o3d_pcd(points_2, os.path.join(dir_path, f"points_target_{i}.pcd"))
        viz_utils.save_o3d_pcd(targets_trans, os.path.join(dir_path, f"points_source_{i}.pcd"))

    for s in sliders + scale_sliders:
        s.on_changed(sliders_on_changed)
    button.on_clicked(button_on_changed)

    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str, help=constants.TASKS_DESCRIPTION)
parser.add_argument("load_path")
main(parser.parse_args())
