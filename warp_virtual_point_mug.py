import argparse
from typing import Tuple, Dict, List, Any, Optional
import pickle
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def warp_object(canonical_obj: NDArray, pca: PCA, latents: NDArray, scale_factor: float):
    return canonical_obj + pca.inverse_transform(latents).reshape((-1, 3)) / scale_factor


def update_axis(ax, new_obj: NDArray, vmin: float, vmax: float, alpha: float, vp: Optional[NDArray]=None):
    """Update 3D plot with a new object point cloud and possibly a new virtual point."""
    ax.clear()
    ax.scatter(new_obj[:, 0], new_obj[:, 1], new_obj[:, 2], color="red", alpha=alpha)
    if vp is not None:
        ax.scatter([vp[0]], [vp[1]], [vp[2]], color="green")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_zlim(vmin, vmax)


def choose_virtual_point(canonical_obj: NDArray, pca: PCA, scale_factor: float) -> NDArray:
    """Choose a virtual point to clone."""
    new_obj = warp_object(canonical_obj, pca, np.array([[0.] * pca.n_components]), scale_factor)
    vp = np.zeros(3, dtype=np.float32)
    # TODO: could be determined from the point cloud
    smin, smax = -0.3, 0.3
    alpha = 0.1

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    update_axis(ax, new_obj, smin, smax, alpha, vp)

    slider_axes = []
    z = 0.
    for _ in range(3):
        slider_axes.append(fig.add_axes([0.25, z, 0.65, 0.03]))
        z += 0.05
    # we start at the bottom and move up
    slider_axes = list(reversed(slider_axes))

    sliders = []
    for i in range(3):
        sliders.append(Slider(slider_axes[i], "D{:d}".format(i), smin, smax, valinit=0))

    def sliders_on_changed(val):
        vp[:] = [s.val for s in sliders]
        update_axis(ax, new_obj, smin, smax, alpha, vp)

    for s in sliders:
        s.on_changed(sliders_on_changed)

    plt.show()
    return vp


def show_anchors(canonical_obj: NDArray, pca: PCA, scale_factor: float, vp: NDArray, knn: NDArray):

    new_obj = warp_object(canonical_obj, pca, np.array([[0.] * pca.n_components]), scale_factor)
    vmin, vmax = -0.3, 0.3
    alpha = 0.1

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    ax.clear()

    colors = np.zeros_like(new_obj)
    colors[:, 0] = 1.
    colors[knn, 0] = 0.
    colors[knn, 2] = 1.

    ax.scatter(new_obj[:, 0], new_obj[:, 1], new_obj[:, 2], color=colors, alpha=alpha)
    if vp is not None:
        ax.scatter([vp[0]], [vp[1]], [vp[2]], color="green")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_zlim(vmin, vmax)

    plt.show()


def warp_virutal_point(canonical_obj: NDArray, pca: PCA, scale_factor: float, vp: NDArray, knn: NDArray, deltas: NDArray):

    new_obj = warp_object(canonical_obj, pca, np.array([[0.] * pca.n_components]), scale_factor)
    smin, smax = -2., 2.
    vmin, vmax = -0.3, 0.3
    alpha = 0.1

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    update_axis(ax, new_obj, vmin, vmax, alpha, vp)

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
        new_obj = warp_object(canonical_obj, pca, latents, scale_factor)
        anchors = new_obj[knn]
        targets = anchors + deltas
        vp = np.mean(targets, axis=0)
        update_axis(ax, new_obj, vmin, vmax, alpha, vp)

    for s in sliders:
        s.on_changed(sliders_on_changed)

    plt.show()


def main(args):

    with open(args.load_path, "rb") as f:
        d = pickle.load(f)
        pca = d["pca"]
        canonical_obj = d["canonical_obj"]

    vp = choose_virtual_point(canonical_obj, pca, args.scale)
    new_obj = warp_object(canonical_obj, pca, np.array([[0.] * pca.n_components]), args.scale)

    k = 10
    dists = np.sum(np.square(new_obj - vp[None]), axis=1)
    knn = np.argpartition(dists, k)[:k]
    print(knn.shape)
    deltas = vp[None] - new_obj[knn]

    show_anchors(canonical_obj, pca, args.scale, vp, knn)
    warp_virutal_point(canonical_obj, pca, args.scale, vp, knn, deltas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_path")
    parser.add_argument("--scale", type=float, default=1.)
    main(parser.parse_args())
