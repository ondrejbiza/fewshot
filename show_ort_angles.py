import numpy as np
import matplotlib.pyplot as plt
import torch

import utils


def main():

    num = 3 * 60
    orts = utils.random_ortho_rots_hemisphere(num)
    rots = utils.orthogonalize(torch.tensor(orts, device="cpu")).numpy()

    x = np.array([[1., 0., 0.]] * num)
    y = np.array([[0., 1., 0.]] * num)
    z = np.array([[0., 0., 1.]] * num)

    out_x = np.einsum("bnk,bkl->bnl", rots, x[:, :, None])[:, :, 0]
    out_y = np.einsum("bnk,bkl->bnl", rots, y[:, :, None])[:, :, 0]
    out_z = np.einsum("bnk,bkl->bnl", rots, z[:, :, None])[:, :, 0]

    fig = plt.figure()
    ax = fig.add_subplot(131, projection="3d")
    ax.scatter(out_x[:, 0], out_x[:, 1], out_x[:, 2])
    ax = fig.add_subplot(132, projection="3d")
    ax.scatter(out_y[:, 0], out_y[:, 1], out_y[:, 2])
    ax = fig.add_subplot(133, projection="3d")
    ax.scatter(out_z[:, 0], out_z[:, 1], out_z[:, 2])
    plt.show()


main()
