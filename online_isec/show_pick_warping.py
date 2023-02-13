import argparse
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pickle

import utils


def update_axis(ax, new_obj: NDArray, index: int, vmin: float, vmax: float):

    ax.clear()
    ax.scatter(new_obj[:, 0], new_obj[:, 1], new_obj[:, 2], color="green", alpha=0.5)
    ax.scatter([new_obj[index, 0]], [new_obj[index, 1]], color="red", s=200)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_zlim(vmin, vmax)


def main(args):

    with open(args.canon_mug_path, "rb") as f:
        canon_mug = pickle.load(f)

    with open(args.load_path, "rb") as f:
        data = pickle.load(f)
        index, target_quat = data["index"], data["quat"]

    new_obj = utils.canon_to_pc(canon_mug, [np.array([[0.] * canon_mug["pca"].n_components])])
    smin, smax = -2., 2.
    vmin, vmax = -0.3, 0.3

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    update_axis(ax, new_obj, index, vmin, vmax)

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
        update_axis(ax, new_obj, index, vmin, vmax)

    for s in sliders:
        s.on_changed(sliders_on_changed)

    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("load_path")
parser.add_argument("--canon-mug-path", default="data/230201_ndf_mugs_large_pca_8_dim.npy")
main(parser.parse_args())
