import copy
from typing import Dict

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import open3d as o3d
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
        x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2], marker={"size": 5, "color": pcd[:, 2], "colorscale": "Plotly3"}, mode="markers", opacity=1.)]
    layout = {
        "xaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "yaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "zaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "aspectratio": {"x": 1, "y": 1, "z": 1}
    }

    fig = go.Figure(data=data)
    fig.update_layout(scene=layout)
    fig.show()
    # input("Continue?")


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
            marker={"size": 5, "color": v[:, 2], "colorscale": colorscale},
            mode="markers", opacity=1., name=key)
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
    # input("Continue?")


def draw_square(img: NDArray, x: int, y: int, square_size=20, copy=False, intensity: float=1.) -> NDArray:
    """Draw square in image."""
    size = square_size // 2
    x_limits = [x - size, x + size]
    y_limits = [y - size, y + size]
    for i in range(len(x_limits)):
        x_limits[i] = min(img.shape[0], max(0, x_limits[i]))
    for i in range(len(y_limits)):
        y_limits[i] = min(img.shape[1], max(0, y_limits[i]))

    if copy:
        img = np.array(img, dtype=img.dtype)

    if img.dtype == np.uint8:
        img[x_limits[0]: x_limits[1], y_limits[0]: y_limits[1]] = int(255 * intensity)
    else:
        img[x_limits[0]: x_limits[1], y_limits[0]: y_limits[1]] = intensity

    return img


def save_o3d_pcd(pcd: NDArray[np.float32], save_path: str):
   
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(save_path, pcd_o3d)
