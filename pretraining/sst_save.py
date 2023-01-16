import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import trimesh
from sst_utils import load_object_create_verts, pick_canonical, cpd_transform, cpd_transform_plot, warp_gen, \
    pca_transform, pca_reconstruct


def main(args):

    trimesh.util.attach_to_log()

    if args.objects == "real_tree":
        obj_path = "data/real_tree.stl"
        scale = 1.
        rotation = Rotation.from_euler("zyx", [0., 0., 0.]).as_matrix()
    else:
        raise ValueError("Unknown object class.")

    obj = load_object_create_verts(obj_path, scale=scale, rotation=rotation)

    if args.show:
        points = obj["points"]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[..., 0], points[..., 1], points[..., 2], color=[1., 0., 0., 1.])
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(-0.2, 0.2)
        plt.show()

    print("# points: {:d}".format(len(obj["points"])))

    with open(args.save_path, "wb") as f:
        pickle.dump({
            "pca": None,
            "canonical_obj": obj["points"],
            "canonical_mesh_points": obj["mesh_points"],
            "canonical_mesh_faces": obj["faces"],
            "scale": args.scale
        }, f)


parser = argparse.ArgumentParser()
parser.add_argument("save_path")
parser.add_argument("objects")
parser.add_argument("--scale", type=float, default=1.)
parser.add_argument("--show", default=False, action="store_true")
main(parser.parse_args())
