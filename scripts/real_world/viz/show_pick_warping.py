import argparse
import os
import pickle

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from src import utils, viz_utils
from src.real_world import constants


def update_axis(ax, source_obj: NDArray, robotiq_points: NDArray, vmin: float, vmax: float):

    ax.clear()
    ax.scatter(source_obj[:, 0], source_obj[:, 1], source_obj[:, 2], color="red", alpha=0.2)
    ax.scatter(robotiq_points[:, 0], robotiq_points[:, 1], robotiq_points[:, 2], color="green", alpha=1.0, s=100)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_zlim(vmin, vmax)


def main(args):

    smin, smax = -2., 2.
    vmin, vmax = -0.3, 0.3

    if args.task == "mug_tree":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_MUGS_PCA_PATH)
        canon_source.init_scale = constants.NDF_MUGS_INIT_SCALE
    elif args.task == "bowl_on_mug":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_BOWLS_PCA_PATH)
        canon_source.init_scale = constants.NDF_BOWLS_INIT_SCALE
    elif args.task == "bottle_in_box":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_BOTTLES_PCA_PATH)
        canon_source.init_scale = constants.NDF_BOTTLES_INIT_SCALE
    else:
        raise NotImplementedError()

    with open(args.load_path, "rb") as f:
        d = pickle.load(f)
    index, pos_robotiq = d["index"], d["pos_robotiq"]

    pcd = canon_source.to_pcd(utils.ObjParam(latent=np.zeros(canon_source.n_components)))

    pos = pcd[index]
    trans, _, _ = utils.best_fit_transform(pos, pos_robotiq)
    pcd = utils.transform_pcd(pcd, trans)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    update_axis(ax, pcd, pos_robotiq, vmin, vmax)

    slider_axes = []
    z = 0.
    for _ in range(canon_source.n_components + 1):
        slider_axes.append(fig.add_axes([0.25, z, 0.65, 0.03]))
        z += 0.05
    # we start at the bottom and move up
    slider_axes = list(reversed(slider_axes))

    sliders = []
    for i in range(canon_source.n_components):
        sliders.append(Slider(slider_axes[i], "D{:d}".format(i), smin, smax, valinit=0))
    button = Button(slider_axes[canon_source.n_components], "Save pcd")

    def sliders_on_changed(val):
        latents = np.array([[s.val for s in sliders]])
        source_pcd = canon_source.to_pcd(utils.ObjParam(latent=latents, scale=np.ones(3) * canon_source.init_scale))
        pos = source_pcd[index]
        trans, _, _ = utils.best_fit_transform(pos, pos_robotiq)
        source_pcd = utils.transform_pcd(source_pcd, trans)
        update_axis(ax, source_pcd, pos_robotiq, vmin, vmax)

    def button_on_changed(val):
        latents = np.array([[s.val for s in sliders]])
        obj_param = utils.ObjParam(latent=latents, scale=np.ones(3) * canon_source.init_scale)

        source_pcd = canon_source.to_pcd(obj_param)
        pos = source_pcd[index]
        trans, _, _ = utils.best_fit_transform(pos, pos_robotiq)
        source_pcd = utils.transform_pcd(source_pcd, trans)
        pos_trans = utils.transform_pcd(pos, trans)

        pos, quat = utils.transform_to_pos_quat(trans)
        obj_param.position = pos
        obj_param.quat = quat
        mesh = canon_source.to_transformed_mesh(obj_param)

        dir_path = "data/warping_figure_4"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        i = 1
        for i in range(1, 1000):
            file_path = os.path.join(dir_path, f"{i}.pcd")
            if not os.path.isfile(file_path):
                break

        mesh.export(os.path.join(dir_path, f"{i}.stl"))
        viz_utils.save_o3d_pcd(source_pcd, os.path.join(dir_path, f"{i}.pcd"))
        viz_utils.save_o3d_pcd(pos_robotiq, os.path.join(dir_path, f"points_robotiq_{i}.pcd"))
        viz_utils.save_o3d_pcd(pos_trans, os.path.join(dir_path, f"points_source_{i}.pcd"))

    for s in sliders:
        s.on_changed(sliders_on_changed)
    button.on_clicked(button_on_changed)

    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str, help=constants.TASKS_DESCRIPTION)
parser.add_argument("load_path")
main(parser.parse_args())
