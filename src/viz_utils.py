import copy
from typing import Dict

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def show_pcd_pyplot(pcd: NDArray, center: bool=False):

    if center:
        pcd = pcd - np.mean(pcd, axis=0, keepdims=True)
    lmin = np.min(pcd)
    lmax = np.max(pcd)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2])
    ax.set_xlim(lmin, lmax)
    ax.set_ylim(lmin, lmax)
    ax.set_zlim(lmin, lmax)
    plt.show()


def show_pcds_pyplot(pcds: Dict[str, NDArray], center: bool=False):

    if center:
        tmp = np.concatenate(list(pcds.values()), axis=0)
        m = np.mean(tmp, axis=0)
        pcds = copy.deepcopy(pcds)
        for k in pcds.keys():
            pcds[k] = pcds[k] - m[None]

    tmp = np.concatenate(list(pcds.values()), axis=0)
    lmin = np.min(tmp)
    lmax = np.max(tmp)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for label, pcd in pcds.items():
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], label=label)
    ax.set_xlim(lmin, lmax)
    ax.set_ylim(lmin, lmax)
    ax.set_zlim(lmin, lmax)
    plt.legend()
    plt.show()
