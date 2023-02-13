import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import open3d as o3d
from sst_utils import load_object_create_verts, pick_canonical, cpd_transform, cpd_transform_plot, warp_gen, \
    pca_transform, pca_reconstruct
import trimesh
import plotly.graph_objects as go


def main(args):

    obj_paths = []

    base_dir = "data/ndf_mug_grid"
    for row in range(4):
        for column in range(5):
            obj_paths.append(os.path.join(base_dir, "{:d}_{:d}.stl".format(row, column)))

    pl_meshes = []

    x_coords = np.linspace(-0.3, 0.3, 5)
    y_coords = np.linspace(0.3, -0.3, 5)

    for i, obj_path in enumerate(obj_paths):

        mesh = trimesh.load(obj_path, force=True)

        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        x_delta = x_coords[i % 5]
        y_delta = y_coords[i // 5]

        rot = Rotation.from_euler("z", np.pi / 6).as_matrix()
        vertices = np.matmul(vertices, rot.T)

        pl_mesh = go.Mesh3d(
            x=vertices[:, 0] + x_delta, y=vertices[:, 1], z=vertices[:, 2] + y_delta,
            i=faces[:, 0], k=faces[:, 1], j=faces[:, 2],
            colorscale="OrRd",
            intensity=vertices[:, 1]
        )
        pl_meshes.append(pl_mesh)

    layout = {
        "xaxis": {"visible": False, "range": [-10, 10]},
        "yaxis": {"visible": False, "range": [-10, 10]},
        "zaxis": {"visible": False, "range": [-10, 10]},
    }

    # camera = {
    #     "up": {"x": 0, "y": 1, "z": 0},
    #     "eye": {"x": 0.0, "y": 0.1, "z": -0.1,},
    #     "center": {"x": 0, "y": 0, "z": 0},
    # }

    fig = go.Figure(data=pl_meshes)
    # fig.update_layout(scene_camera=camera)
    fig.update_layout(scene=layout)
    fig.show()


parser = argparse.ArgumentParser()
main(parser.parse_args())
