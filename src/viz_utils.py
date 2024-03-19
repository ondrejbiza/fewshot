# edited from ondrej_biza/fewshot

import copy
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go


def show_pcd_pyplot(pcd: NDArray, center: bool = False):
    print("VIZUALIZING")
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


def show_pcd_plotly(pcd: NDArray, center: bool = False, axis_visible: bool = True):
    if center:
        pcd = pcd - np.mean(pcd, axis=0, keepdims=True)
    lmin = np.min(pcd)
    lmax = np.max(pcd)

    data = [
        go.Scatter3d(
            x=pcd[:, 0],
            y=pcd[:, 1],
            z=pcd[:, 2],
            marker={"size": 5, "color": pcd[:, 2], "colorscale": "Plotly3"},
            mode="markers",
            opacity=1.0,
        )
    ]
    layout = {
        "xaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "yaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "zaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "aspectratio": {"x": 1, "y": 1, "z": 1},
    }

    fig = go.Figure(data=data)
    fig.update_layout(scene=layout)
    fig.show()
    return fig


def show_pcds_pyplot(pcds: Dict[str, NDArray], center: bool = False):
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


def show_pcds_plotly(
    pcds: Dict[str, NDArray], center: bool = False, axis_visible: bool = True
):
    colorscales = [
        "Plotly3",
        "Viridis",
        "Blues",
        "Greens",
        "Greys",
        "Oranges",
        "Purples",
        "Reds",
    ]

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
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            marker={"size": 5, "color": v[:, 2], "colorscale": colorscale},
            mode="markers",
            opacity=1.0,
            name=key,
        )
        data.append(pl)

    layout = {
        "xaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "yaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "zaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "aspectratio": {"x": 1, "y": 1, "z": 1},
    }
    fig = go.Figure(data=data)
    fig.update_layout(scene=layout, showlegend=True)
    fig.show()
    return fig


def show_meshes_plotly(
    vertices: Dict[str, NDArray],
    faces: Dict[str, NDArray],
    center: bool = False,
    axis_visible: bool = True,
):
    colorscales = [
        "Plotly3",
        "Viridis",
        "Blues",
        "Greens",
        "Greys",
        "Oranges",
        "Purples",
        "Reds",
    ]

    if center:
        tmp = np.concatenate(list(vertices.values()), axis=0)
        m = np.mean(tmp, axis=0)
        vertices = copy.deepcopy(vertices)
        for k in vertices.keys():
            vertices[k] = vertices[k] - m[None]

    tmp = np.concatenate(list(vertices.values()), axis=0)
    lmin = np.min(tmp)
    lmax = np.max(tmp)

    data = []
    for idx, key in enumerate(vertices.keys()):
        v = vertices[key]
        f = faces[key]
        colorscale = colorscales[idx % len(colorscales)]
        mesh = go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=f[:, 0],
            j=f[:, 1],
            k=f[:, 2],
            colorscale=colorscale,
            intensity = v[:, 2],
        )
        data.append(mesh)

    layout = {
        "xaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "yaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "zaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "aspectratio": {"x": 1, "y": 1, "z": 1},
    }
    fig = go.Figure(data=data)
    fig.update_layout(scene=layout, showlegend=True)
    fig.show()
    return fig


from plotly.subplots import make_subplots


def show_pcd_grid_plotly(
    rows,
    cols,
    pcls: Dict[str, NDArray],
    names: List[str],
    colorscales: Optional[List[str]] = None,
    camera_views: Optional[List[Dict[str, dict]]] = None,
    save_image: bool = False,
    save_path: bool = False,
):
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scatter3d"} for i in range(cols)] for j in range(rows)],
    )

    for row in range(rows):
        for col in range(cols):
            if colorscales is not None:
                colorscale = colorscales[row * cols + col]
            else:
                colorscale = "viridis"
            name = names[rows * cols + col]

            fig.add_trace(
                go.Scatter3d(
                    x=pcls[name][:, 0],
                    y=pcls[name][:, 1],
                    z=pcls[name][:, 2],
                    marker={
                        "size": 5,
                        "color": pcls[name][:, 0],
                        "colorscale": colorscale,
                    },
                    mode="markers",
                    opacity=1.0,
                    name=name,
                ),
                row=row + 1,
                col=col + 1,
            )
    fw = go.FigureWidget(fig)

    if camera_views is not None:
        all_cameras = [
            eval(f"fw.layout.scene{i}.camera") for i in range(1, row * cols + 1, 1)
        ]

        with fw.batch_update():
            fw.layout.update(width=800, height=600)
            for i in range(len(all_cameras)):
                camera = all_cameras[i]
                camera.up = camera_views[i]["up"]  # dict(x=0, y=1, z=0)
                camera.eye = camera_views[i]["eye"]  # dict(x=2.5, y=1.75, z=1)

        fw.update_layout(height=1000, width=1000)
    if save_image:
        fw.write_image(save_path)

    fw.show()
    return fw


import moviepy.editor as mpy
import io
import copy as cp
from PIL import Image


def plotly_fig2array(fig):
    # convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def show_pcds_video_animation_plotly(
    moving_pcl_name: str,
    moving_pcl_frames: NDArray,
    static_pcls: Dict[str, NDArray],
    step_names: List[str],
    file_name: str,
):
    video_length = 2

    visible = [False] * (len(moving_pcl_frames)) + [True] * len(static_pcls.keys())

    fig = go.Figure()

    # Add traces, one for each slider step
    for t, moving_pcl_frame in enumerate(moving_pcl_frames):
        fig.add_trace(
            go.Scatter3d(
                visible=False,
                x=moving_pcl_frame[:, 0],
                y=moving_pcl_frame[:, 1],
                z=moving_pcl_frame[:, 2],
                marker={
                    "size": 5,
                    "color": moving_pcl_frame[:, 2],
                    "colorscale": "viridis",
                },
                mode="markers",
                opacity=1.0,
                name=f"{moving_pcl_name}, Step {t}",
            ),
        )

    for static_name in static_pcls.keys():
        fig.add_trace(
            go.Scatter3d(
                visible=True,
                x=static_pcls[static_name][:, 0],
                y=static_pcls[static_name][:, 1],
                z=static_pcls[static_name][:, 2],
                marker={
                    "size": 5,
                    "color": static_pcls[static_name][:, 2],
                    "colorscale": "plotly3",
                },
                mode="markers",
                opacity=1.0,
                name=static_name,
            ),
        )

    def make_frame(t):
        # z = f(2*np.pi*t/2)
        i = int(t * len(moving_pcl_frames))

        for j in range(len(moving_pcl_frames)):
            fig.update_traces(
                visible=False, selector=dict(name=f"{moving_pcl_name}, Step {j}")
            )

        fig.update_traces(
            visible=True, selector=dict(name=f"{moving_pcl_name}, Step {i}")
        )  # These are the updates that usually are performed within Plotly go.Frame definition
        fig.update_layout(title=step_names[i])
        return plotly_fig2array(fig)

    animation = mpy.VideoClip(make_frame, duration=1)
    animation.write_gif(f"{file_name}.gif", fps=20)


def show_pcds_slider_animation_plotly(
    moving_pcl_name: str,
    moving_pcl_frames: NDArray,
    static_pcls: Dict[str, NDArray],
    step_names: List[str],
):
    fig = go.Figure()

    # Add traces, one for each slider step
    for t, moving_pcl_frame in enumerate(moving_pcl_frames):
        fig.add_trace(
            go.Scatter3d(
                visible=False,
                x=moving_pcl_frame[:, 0],
                y=moving_pcl_frame[:, 1],
                z=moving_pcl_frame[:, 2],
                marker={
                    "size": 5,
                    "color": moving_pcl_frame[:, 2],
                    "colorscale": "viridis",
                },
                mode="markers",
                opacity=1.0,
                name=f"moving_pcl_name_{t}",
            ),
        )

    for static_name in static_pcls.keys():
        fig.add_trace(
            go.Scatter3d(
                x=static_pcls[static_name][:, 0],
                y=static_pcls[static_name][:, 1],
                z=static_pcls[static_name][:, 2],
                marker={
                    "size": 5,
                    "color": static_pcls[static_name][:, 2],
                    "colorscale": "plotly3",
                },
                mode="markers",
                opacity=1.0,
                name=static_name,
            ),
        )

    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data) - 1):
        step = dict(
            method="update",
            args=[
                {
                    "visible": [False] * (len(moving_pcl_frames))
                    + [True] * len(static_pcls.keys())
                },
                {"title": step_names[i]},
            ],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [
        dict(active=10, currentvalue={"prefix": "Step: "}, pad={"t": 50}, steps=steps)
    ]

    fig.update_layout(sliders=sliders)
    fig.show()
    return fig


def draw_square(
    img: NDArray, x: int, y: int, square_size=20, copy=False, intensity: float = 1.0
) -> NDArray:
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
        img[x_limits[0] : x_limits[1], y_limits[0] : y_limits[1]] = int(255 * intensity)
    else:
        img[x_limits[0] : x_limits[1], y_limits[0] : y_limits[1]] = intensity

    return img


def save_o3d_pcd(pcd: NDArray[np.float32], save_path: str):
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(save_path, pcd_o3d)


def draw_arrow(ax, orig, delta, color):
    ax.quiver(
        orig[0],
        orig[1],
        orig[2],  # <-- starting point of vector
        delta[0],
        delta[1],
        delta[2],  # <-- directions of vector
        color=color,
        alpha=0.8,
        lw=3,
    )


def show_pose(ax, T):
    orig = T[:3, 3]
    rot = T[:3, :3]
    x_arrow = np.matmul(rot, np.array([0.05, 0.0, 0.0]))
    y_arrow = np.matmul(rot, np.array([0.0, 0.05, 0.0]))
    z_arrow = np.matmul(rot, np.array([0.0, 0.0, 0.05]))
    draw_arrow(ax, orig, x_arrow, "red")
    draw_arrow(ax, orig, y_arrow, "green")
    draw_arrow(ax, orig, z_arrow, "blue")
