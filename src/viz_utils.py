import copy
from typing import Dict

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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


def show_pcd_plotly(pcd: NDArray, center: bool=False, axis_visible: bool=True):

    if center:
        pcd = pcd - np.mean(pcd, axis=0, keepdims=True)
    lmin = np.min(pcd)
    lmax = np.max(pcd)

    data = [go.Scatter3d(
        x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2], marker={"size": 10, "color": pcd[:, 2], "colorscale": "Plotly3"}, mode="markers", opacity=0.9)]
    layout = {
        "xaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "yaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "zaxis": {"visible": axis_visible, "range": [lmin, lmax]},
    }

    fig = go.Figure(data=data)
    fig.update_layout(scene=layout)
    fig.show()
    input("Continue?")


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


def show_pcds_plotly(pcds: Dict[str, NDArray], center: bool=False, axis_visible: bool=True):

    colorscales = ["Plotly3", "Viridis", "Blues", "Greens", "Greys", "Oranges", "Purples", "Reds"]

    if center:
        tmp = np.concatenate(list(pcds.values()), axis=0)
        m = np.mean(tmp, axis=0)
        pcds = copy.deepcopy(pcds)
        for k in pcds.keys():
            pcds[k] = pcds[k] - m[None]

    tmp = np.concatenate(list(pcds.values()), axis=0)
    lmin = np.min(tmp)
    lmax = np.max(tmp)

    data = []
    for idx, key in enumerate(pcds.keys()):
        v = pcds[key]
        colorscale = colorscales[idx % len(colorscales)]
        pl = go.Scatter3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            marker={"size": 10, "color": v[:, 2], "colorscale": colorscale},
            mode="markers", opacity=0.9, name=key)
        data.append(pl)

    layout = {
        "xaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "yaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "zaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "aspectratio": {"x": 1, "y": 1, "z": 1}
    }

    fig = go.Figure(data=data)
    fig.update_layout(scene=layout, showlegend=True)
    fig.show()
    input("Continue?")
