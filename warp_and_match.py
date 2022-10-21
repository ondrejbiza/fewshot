import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle
import utils


def interpolate_pca(pca, canonical_obj, indices):

    n_components = pca.n_components
    new_obj = canonical_obj + pca.inverse_transform(np.array([[0.] * n_components])).reshape((-1, 3)) / 2.

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    colors = np.zeros_like(new_obj)
    colors[:, 1] = 1.
    colors[indices, 1] = 0.
    colors[indices, 0] = 1.
    ax.scatter(new_obj[:, 0],  new_obj[:, 1], new_obj[:, 2], color=colors)
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)

    slider_axes = []
    z = 0.
    for _ in range(n_components):
        slider_axes.append(fig.add_axes([0.25, z, 0.65, 0.03]))
        z += 0.05
    # we start at the bottom and move up
    slider_axes = list(reversed(slider_axes))

    smin, smax = -2., 2.

    sliders = []
    for i in range(n_components):
        sliders.append(Slider(slider_axes[i], 'D{:d}'.format(i), smin, smax, valinit=0))

    def sliders_on_changed(val):
        new_obj = canonical_obj + pca.inverse_transform(
            np.array([[s.val for s in sliders]])
        ).reshape((-1, 3))
        ax.clear()
        colors[:, 1] = 1.
        colors[indices, 1] = 0.
        colors[indices, 0] = 1.
        ax.scatter(new_obj[:, 0], new_obj[:, 1], new_obj[:, 2], color=colors)
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(-0.3, 0.3)

    for s in sliders:
        s.on_changed(sliders_on_changed)

    plt.show()
    return np.array([s.val for s in sliders])


def main(args):

    indices = np.load(args.load_path)

    canon = {}
    with open("data/mugs_pca.pkl", "rb") as f:
        canon[1] = pickle.load(f)
    with open("data/trees_pca_8d.pkl", "rb") as f:
        canon[2] = pickle.load(f)

    latent_1 = interpolate_pca(canon[1]["pca"], canon[1]["canonical_obj"], indices[0])
    print("latent 1:", latent_1)
    latent_2 = interpolate_pca(canon[2]["pca"], canon[2]["canonical_obj"], indices[1])
    print("latent 2:", latent_2)

    new_obj_1 = canon[1]["canonical_obj"] + canon[1]["pca"].inverse_transform(np.array([latent_1])).reshape((-1, 3)) / 2.
    new_obj_2 = canon[2]["canonical_obj"] + canon[2]["pca"].inverse_transform(np.array([latent_2])).reshape((-1, 3)) / 2.

    points_1 = new_obj_1[indices[0]]
    points_2 = new_obj_2[indices[1]]

    print("points to match 1:", points_1)
    print("points to match 2:", points_2)

    T, R, t = utils.best_fit_transform(points_1, points_2)

    tmp = np.concatenate([new_obj_1, np.ones_like(new_obj_1)[..., -1:]], axis=-1)
    tmp = np.matmul(T, tmp.T).T
    tmp /= tmp[..., -1][..., None]
    tmp = tmp[..., :-1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(new_obj_2[:, 0], new_obj_2[:, 1], new_obj_2[:, 2])
    ax.scatter(tmp[:, 0], tmp[:, 1], tmp[:, 2])
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("load_path")
main(parser.parse_args())
