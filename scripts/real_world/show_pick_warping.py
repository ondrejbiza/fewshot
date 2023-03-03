import argparse
import pickle

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from src import utils


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

    if args.task == "bowl_on_mug":
        canon_source = utils.CanonObj.from_pickle("data/230227_ndf_bowls_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_source.init_scale = 0.8
    else:
        raise NotImplementedError()

    with open(args.load_path, "rb") as f:
        d = pickle.load(f)
    index, pos_robotiq = d["index"], d["pos_robotiq"]

    pcd = canon_source.to_pcd(utils.ObjParam(latent=np.zeros(canon_source.n_components)))

    pos = pcd[index]
    trans, _, _ = utils.best_fit_transform(pos_robotiq, pos)
    pos_target = utils.transform_pcd(pos_robotiq, trans)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    update_axis(ax, pcd, pos_target, vmin, vmax)

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
        source_pcd = canon_source.to_pcd(utils.ObjParam(latent=latents, scale=np.ones(3) * canon_source.init_scale))
        pos = source_pcd[index]
        trans, _, _ = utils.best_fit_transform(pos_robotiq, pos)
        pos_target = utils.transform_pcd(pos_robotiq, trans)
        update_axis(ax, source_pcd, pos_target, vmin, vmax)

    for s in sliders:
        s.on_changed(sliders_on_changed)

    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str, help="[mug_tree, bowl_on_mug, bottle_in_box]")
parser.add_argument("load_path")
main(parser.parse_args())