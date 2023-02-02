import argparse
import pickle
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import online_isec.utils as isec_utils
from online_isec.show_ndf_demonstration import show_pose


def update_axis(ax, mug_pc: NDArray, tree_pc: NDArray, T1: NDArray, T2: NDArray, point: NDArray):

    ax.clear()

    ax.scatter(mug_pc[:, 0], mug_pc[:, 1], mug_pc[:, 2], alpha=0.5, c="blue")
    ax.scatter(tree_pc[:, 0], tree_pc[:, 1], tree_pc[:, 2], alpha=0.5, c="brown")

    ax.scatter([point[0]], [point[1]], [point[2]], c="red", s=100)

    show_pose(ax, T1)
    show_pose(ax, T2)

    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(0., 0.4)


def main(args):

    T_ws_to_b = isec_utils.workspace_to_base()

    with open(args.pick_load_path, "rb") as f:
        x = pickle.load(f)
        mug_pc = x["observed_pc"]
        pick_T_g_to_b = x["T_g_to_b"]

    with open(args.place_load_path, "rb") as f:
        x = pickle.load(f)
        tree_pc = x["observed_pc"]
        place_T_g_to_b = x["T_g_to_b"] 
        place_T_g_pre_to_g = x["T_g_pre_to_g"] 

    place_T_g_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), place_T_g_to_b)

    pick_T_g_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), pick_T_g_to_b)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    update_axis(ax, mug_pc, tree_pc, pick_T_g_to_ws, place_T_g_to_ws, np.array([0., 0., 0.]))

    slider_axes = []
    z = 0.
    for _ in range(3):
        slider_axes.append(fig.add_axes([0.25, z, 0.65, 0.03]))
        z += 0.05
    slider_axes = list(reversed(slider_axes))

    sliders = []
    sliders.append(Slider(slider_axes[0], "D{:d}".format(0), -0.2, 0.2, valinit=0))
    sliders.append(Slider(slider_axes[1], "D{:d}".format(1), -0.2, 0.2, valinit=0))
    sliders.append(Slider(slider_axes[2], "D{:d}".format(2), 0., 0.4, valinit=0))

    def sliders_on_changed(val):
        point = np.array([s.val for s in sliders])
        update_axis(ax, mug_pc, tree_pc, pick_T_g_to_ws, place_T_g_to_ws, point)

    for s in sliders:
        s.on_changed(sliders_on_changed)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pick_load_path")
    parser.add_argument("place_load_path")
    main(parser.parse_args())
