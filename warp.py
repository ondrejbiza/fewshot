import argparse
import pickle
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def warp_object(canonical_obj: NDArray, pca: PCA, latents: NDArray, scale_factor: float):
    return canonical_obj + pca.inverse_transform(latents).reshape((-1, 3)) / scale_factor


def update_axis(ax, new_obj: NDArray, vmin: float, vmax: float):

    ax.clear()
    ax.scatter(new_obj[:, 0], new_obj[:, 1], new_obj[:, 2], color="red")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_zlim(vmin, vmax)


def main(args):

    with open(args.load_path, "rb") as f:
        d = pickle.load(f)
        pca = d["pca"]
        canonical_obj = d["canonical_obj"]

    new_obj = warp_object(canonical_obj, pca, np.array([[0.] * pca.n_components]), args.scale)
    smin, smax = -2., 2.
    vmin, vmax = -0.3, 0.3

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    update_axis(ax, new_obj, vmin, vmax)

    slider_axes = []
    z = 0.
    for _ in range(pca.n_components):
        slider_axes.append(fig.add_axes([0.25, z, 0.65, 0.03]))
        z += 0.05
    # we start at the bottom and move up
    slider_axes = list(reversed(slider_axes))

    sliders = []
    for i in range(pca.n_components):
        sliders.append(Slider(slider_axes[i], "D{:d}".format(i), smin, smax, valinit=0))

    def sliders_on_changed(val):
        latents = np.array([[s.val for s in sliders]])
        new_obj = warp_object(canonical_obj, pca, latents, args.scale)
        update_axis(ax, new_obj, vmin, vmax)

    for s in sliders:
        s.on_changed(sliders_on_changed)

    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("load_path")
parser.add_argument("--scale", type=float, default=1.)
main(parser.parse_args())