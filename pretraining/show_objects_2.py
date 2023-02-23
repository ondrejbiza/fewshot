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

    example_pc = np.load("data/example_mug_pc.npy")
    example_pc -= np.mean(example_pc, axis=0, keepdims=True)

    if args.objects == "ndf_mugs":
        base_dir = "data/ndf_objects/mug_centered_obj_normalized"
        obj_ids = [
            "1a97f3c83016abca21d0de04f408950f", "1c9f9e25c654cbca3c71bf3f4dd78475", "1eaf8db2dd2b710c7d5b1b70ae595e60",
            "3d1754b7cb46c0ce5c8081810641ef6", "4b8b10d03552e0891898dfa8eb8eefff", "4b7888feea81219ab5f4a9188bfa0ef6",
            "5c48d471200d2bf16e8a121e6886e18d", "5d72df6bc7e93e6dd0cd466c08863ebd", "5fe74baba21bba7ca4eec1b19b3a18f8",
            "6aec84952a5ffcf33f60d03e1cb068dc"]
        obj_paths = [os.path.join(base_dir, x, "models/model_normalized.obj") for x in obj_ids]
        scale = 0.14
        rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()
    elif args.objects == "ndf_bowls":
        base_dir = "data/ndf_objects/bowl_centered_obj_normalized"
        obj_ids = [
            "1b4d7803a3298f8477bdcb8816a3fac9", "1fbb9f70d081630e638b4be15b07b442",
            "2a1e9b5c0cead676b8183a4a81361b94", "2c1df84ec01cea4e525b133235812833",
            "4b32d2c623b54dd4fe296ad57d60d898", "4eefe941048189bdb8046e84ebdc62d2",
            "4fdb0bd89c490108b8c8761d8f1966ba", "5b6d840652f0050061d624c546a68fec",
            "5bb12905529c85359d3d767e1bc88d65", "7c43116dbe35797aea5000d9d3be7992"
        ]
        obj_paths = [os.path.join(base_dir, x, "models/model_normalized.obj") for x in obj_ids]
        scale = 0.14
        rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()
    elif args.objects == "ndf_bottles":
        base_dir = "data/ndf_objects/bottle_centered_obj_normalized"
        obj_ids = [
            "1ae823260851f7d9ea600d1a6d9f6e07", "1b64b36bf7ddae3d7ad11050da24bb12",
            "1c38ca26c65826804c35e2851444dc2f", "1cf98e5b6fff5471c8724d5673a063a6",
            "1d4480abe9aa45ce51a99c0e19a8a54", "1df41477bce9915e362078f6fc3b29f5",
            "1e5abf0465d97d826118a17db9de8c0", "1ef68777bfdb7d6ba7a07ee616e34cd7",
            "1ffd7113492d375593202bf99dddc268", "2a3e0c1cd0e9076cddf5870150a75bc"
        ]
        obj_paths = [os.path.join(base_dir, x, "models/model_normalized.obj") for x in obj_ids]
        scale = 0.14
        rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()
    else:
        raise ValueError("Unknown object class.")

    pl_meshes = []

    x_coords = [-1.2, -0.6, 0., 0.6, 1.2]

    for i, obj_path in enumerate(obj_paths):

        mesh = trimesh.load(obj_path, force=True)

        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        x_delta = x_coords[i % 5]
        if i < 5:
            y_delta = - 0.4
        else:
            y_delta = 0.4

        # rot = Rotation.from_euler("y", np.pi / 6).as_matrix()
        # vertices = np.matmul(vertices, rot.T)

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

    camera = {
        "up": {"x": 0, "y": 1, "z": 0},
        "eye": {"x": 0.0, "y": 0.1, "z": -0.1,},
        "center": {"x": 0, "y": 0, "z": 0},
    }

    fig = go.Figure(data=pl_meshes)
    fig.update_layout(scene_camera=camera)
    fig.update_layout(scene=layout)
    fig.show()


parser = argparse.ArgumentParser()
parser.add_argument("objects")
main(parser.parse_args())
