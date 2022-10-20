import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import trimesh
from sst_utils import load_object_create_verts, pick_canonical, cpd_transform, cpd_transform_plot, warp_gen, \
    pca_transform, pca_reconstruct


def main(args):

    with open(args.load_path, "rb") as f:
        d = pickle.load(f)
        pca = d["pca"]
        canonical_obj = d["canonical_obj"]

    probs = None
    if args.load_probs_path is not None:
        probs = np.load(args.load_probs_path)

    new_obj = canonical_obj + pca.inverse_transform(np.array([[0., 0., 0., 0.]])).reshape((-1, 3)) / 2.

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    color = np.zeros_like(new_obj)
    color[:, 1] = 1.
    if probs is not None:
        color[:, 1] = 1 - np.clip(probs * 2., 0, 1)
        color[:, 0] = np.clip(probs * 2., 0, 1)
    ax.scatter(new_obj[:, 0],  new_obj[:, 1], new_obj[:, 2], color=color)
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)

    d1_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    d2_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    d3_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    d4_ax = fig.add_axes([0.25, 0.0, 0.65, 0.03])

    smin, smax = -2., 2.

    d1_slider = Slider(d1_ax, 'D1', smin, smax, valinit=0)
    d2_slider = Slider(d2_ax, 'D2', smin, smax, valinit=0)
    d3_slider = Slider(d3_ax, 'D3', smin, smax, valinit=0)
    d4_slider = Slider(d4_ax, 'D4', smin, smax, valinit=0)

    def sliders_on_changed(val):
        new_obj = canonical_obj + pca.inverse_transform(
            np.array([[d1_slider.val, d2_slider.val, d3_slider.val, d4_slider.val]])
        ).reshape((-1, 3)) / args.scale
        ax.clear()
        color = np.zeros_like(new_obj)
        color[:, 1] = 1.
        if probs is not None:
            color[:, 1] = 1 - np.clip(probs * 2., 0, 1)
            color[:, 0] = np.clip(probs * 2., 0, 1)
        ax.scatter(new_obj[:, 0], new_obj[:, 1], new_obj[:, 2], color=color)
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(-0.3, 0.3)

    d1_slider.on_changed(sliders_on_changed)
    d2_slider.on_changed(sliders_on_changed)
    d3_slider.on_changed(sliders_on_changed)
    d4_slider.on_changed(sliders_on_changed)

    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("load_path")
parser.add_argument("--scale", type=float, default=1.)
parser.add_argument("-p", "--load-probs-path")
main(parser.parse_args())
