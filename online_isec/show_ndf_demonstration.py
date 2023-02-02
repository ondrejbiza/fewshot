import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

import online_isec.utils as isec_utils


def draw_arrow(ax, orig, delta, color):

    ax.quiver(
        orig[0], orig[1], orig[2], # <-- starting point of vector
        delta[0], delta[1], delta[2], # <-- directions of vector
        color=color, alpha=0.8, lw=3,
    )


def show_pose(ax, T):

    orig = T[:3, 3]
    rot = T[:3, :3]
    x_arrow = np.matmul(rot, np.array([0.05, 0., 0.]))
    y_arrow = np.matmul(rot, np.array([0., 0.05, 0.]))
    z_arrow = np.matmul(rot, np.array([0., 0., 0.05]))
    draw_arrow(ax, orig, x_arrow, "red")
    draw_arrow(ax, orig, y_arrow, "green")
    draw_arrow(ax, orig, z_arrow, "blue")


def main(args):

    with open(args.pick_load_path, "rb") as f:
        x = pickle.load(f)
        mug_pc = x["observed_pc"]
        pick_T_g_to_b = x["T_g_to_b"] 

    with open(args.place_load_path, "rb") as f:
        x = pickle.load(f)
        tree_pc = x["observed_pc"]
        place_T_g_to_b = x["T_g_to_b"] 
        place_T_g_pre_to_g = x["T_g_pre_to_g"] 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # point clouds
    ax.scatter(mug_pc[:, 0], mug_pc[:, 1], mug_pc[:, 2], c="blue")
    ax.scatter(tree_pc[:, 0], tree_pc[:, 1], tree_pc[:, 2], c="brown")

    T_ws_to_b = isec_utils.workspace_to_base()

    # pick pose
    pick_T_g_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), pick_T_g_to_b)
    show_pose(ax, pick_T_g_to_ws)

    # sampled points
    num_samples = 500
    #sigma = 0.015
    sigma = 0.025
    reference_points = np.random.normal(pick_T_g_to_ws[:3, 3], sigma, size=(num_samples, 3))
    ax.scatter(reference_points[:, 0], reference_points[:, 1], reference_points[:, 2], c="gray")

    # pre-place pose
    place_T_g_pre_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), np.matmul(place_T_g_to_b, place_T_g_pre_to_g))
    show_pose(ax, place_T_g_pre_to_ws)

    # place pose
    place_T_g_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), place_T_g_to_b)
    show_pose(ax, place_T_g_to_ws)

    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(0., 0.4)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pick_load_path")
    parser.add_argument("place_load_path")
    main(parser.parse_args())
