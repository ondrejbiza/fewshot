import argparse
import os
import pickle
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import trimesh
import utils


def main(args):

    canonical_mesh_points = None
    canonical_mesh_faces = None
    with open(args.load_path, "rb") as f:
        d = pickle.load(f)
        pca = d["pca"]
        canonical_mesh_points = d["canonical_mesh_points"]
        canonical_mesh_faces = d["canonical_mesh_faces"]

    save_path = "data/ndf_mug_grid"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    values = np.linspace(-1, 1, 5)
    for row in range(4):
        for column in range(5):
            latent = np.zeros(pca.n_components)
            latent[row] = values[column]
            points = utils.canon_to_pc(d, [latent])
            points = points[:len(canonical_mesh_points)]

            mesh = trimesh.Trimesh(vertices=points, faces=canonical_mesh_faces)
            mesh.export(os.path.join(save_path, "{:d}_{:d}.stl".format(row, column)))


parser = argparse.ArgumentParser()
parser.add_argument("load_path")
main(parser.parse_args())
