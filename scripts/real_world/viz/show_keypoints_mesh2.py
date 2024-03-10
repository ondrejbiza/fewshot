import argparse
import os
import pickle

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import pyvista as pv

from src import utils, viz_utils
from src.real_world import constants


def update_axis(ax, source_obj: NDArray, robotiq_points: NDArray, vmin: float, vmax: float):

    ax.clear()
    ax.scatter(source_obj[:, 0], source_obj[:, 1], source_obj[:, 2], color="red", alpha=0.2)
    ax.scatter(robotiq_points[:, 0], robotiq_points[:, 1], robotiq_points[:, 2], color="green", alpha=1.0, s=100)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_zlim(vmin, vmax)


def main(args):

    if args.task == "mug_tree":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_MUGS_PCA_PATH)
        canon_source.init_scale = constants.NDF_MUGS_INIT_SCALE
    elif args.task == "bowl_on_mug":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_BOWLS_PCA_PATH)
        canon_source.init_scale = constants.NDF_BOWLS_INIT_SCALE
    elif args.task == "bottle_in_box":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_BOTTLES_PCA_PATH)
        canon_source.init_scale = constants.NDF_BOTTLES_INIT_SCALE
    else:
        raise NotImplementedError()

    load_path = "data/230330/bowl_on_mug_pick.pkl"
    with open(load_path, "rb") as f:
        x = pickle.load(f)
        grasp_index = x["index"]

    app = Dash(__name__)

    app.layout = html.Div([
        html.H4('PLY Object Explorer'),
        html.P("Choose an object:"),
        dcc.Slider(
            min=-2.0,
            max=2.0,
            step=0.1,
            value=0.0,
            id="axis1",
        ),
        dcc.Slider(
            min=-2.0,
            max=2.0,
            step=0.1,
            value=0.0,
            id="axis2",
        ),
        dcc.Slider(
            min=-2.0,
            max=2.0,
            step=0.1,
            value=0.0,
            id="axis3",
        ),
        dcc.Slider(
            min=-2.0,
            max=2.0,
            step=0.1,
            value=0.0,
            id="axis4",
        ),
        dcc.Graph(id="graph", style={'width': '60vh', 'height': '60vh'}),
    ])

    def ms(x, y, z, radius, resolution=20):
        """Return the coordinates for plotting a sphere centered at (x,y,z)"""
        u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
        X = radius * np.cos(u)*np.sin(v) + x
        Y = radius * np.sin(u)*np.sin(v) + y
        Z = radius * np.cos(v) + z
        return (X, Y, Z)

    def make_sphere(center, radius):

        # center 
        x0 = center[0]
        y0 = center[1]
        z0 = center[2]
        # radius 
        r = radius
        # radius + 2% to be sure of the bounds
        R = r * 1.02

        def f(x, y, z):
            return (
                (x-x0)**2 + (y-y0)**2 + (z-z0)**2
            )

        # generate data grid for computing the values
        X, Y, Z = np.mgrid[(-R+x0):(R+x0):50j, (-R+y0):(R+y0):50j, (-R+z0):(R+z0):50j]
        # create a structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        # compute and assign the values
        values = f(X, Y, Z)
        grid.point_data["values"] = values.ravel(order = "F")
        # compute the isosurface f(x, y, z) = r²
        isosurf = grid.contour(isosurfaces = [r**2])
        mesh = isosurf.extract_geometry()
        # extract vertices and triangles
        vertices = mesh.points
        triangles = mesh.faces.reshape(-1, 4)

        # plot
        return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            # color="lightpink",
            colorscale = [[0, 'rgba(0.1,0,0,1)'],
                        # [0.5, 'mediumturquoise'],
                        [1, 'rgba(1.0,0,0,1)']],
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity = np.linspace(0, 1, len(vertices)),
            # i, j and k give the vertices of the triangles
            i = triangles[:, 1],
            j = triangles[:, 2],
            k = triangles[:, 3],
            showscale = False
        )

    @app.callback(
        Output("graph", "figure"), 
        Input("axis1", "value"),
        Input("axis2", "value"),
        Input("axis3", "value"),
        Input("axis4", "value"))

    def display_mesh(axis1, axis2, axis3, axis4):

        latent = np.zeros(canon_source.n_components)
        latent[0] = axis1
        latent[1] = axis2
        latent[2] = axis3
        latent[3] = axis4

        pcd = canon_source.to_transformed_pcd(utils.ObjParam(latent=latent))
        mesh = canon_source.to_transformed_mesh(utils.ObjParam(latent=latent))

        vertices = mesh.vertices
        faces = mesh.faces

        color = np.zeros((len(faces), 3), dtype=np.float32)
        color[:, 1] = np.linspace(0.0, 0.5, len(faces))
        color[:, 2] = np.linspace(0.0, 1.0, len(faces))

        data = [
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], 
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                facecolor=color,
            ),
            make_sphere(pcd[grasp_index[0]], 0.01),
        ]

        fig = go.Figure(data=data)

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=0.0, z=1)
        )

        fig.update_layout(#plot_bgcolor='rgb(12,163,135)',
                      #paper_bgcolor='rgb(12,163,135)'
                      #coloraxis={"colorbar": {"x": -0.2, "len": 0.5, "y": 0.8}}, #I think this is for contours
                     scene = dict(
                                    xaxis = dict(
                                         title="",
                                         backgroundcolor="rgba(0, 0, 0,0)",
                                         gridcolor="white",
                                         showbackground=True,
                                         showgrid=False,
                                         showticklabels=False,
                                         zerolinecolor="white",),
                                    yaxis = dict(
                                        title="",
                                        backgroundcolor="rgba(0, 0, 0,0)",
                                        gridcolor="white",
                                        showbackground=True,
                                        showgrid=False,
                                        showticklabels=False,
                                        zerolinecolor="white"),
                                    zaxis = dict(
                                        title="",
                                        backgroundcolor="rgba(0, 0, 0,0)",
                                        gridcolor="white",
                                        showbackground=True,
                                        showgrid=False,
                                        showticklabels=False,
                                        zerolinecolor="white",),),
                            scene_camera=camera,
                     )

        return fig

    app.run_server(debug=True)


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str, help=constants.TASKS_DESCRIPTION)
main(parser.parse_args())
