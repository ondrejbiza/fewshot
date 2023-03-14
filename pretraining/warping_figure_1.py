import argparse
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from sst_utils import load_object_create_verts, pick_canonical, warp_gen, pca_transform, scale_object_circle
from src import viz_utils


def main(args):

    np.random.seed(2023)
    trimesh.util.attach_to_log()

    if args.objects == "ndf_mugs":
        base_dir = "data/ndf_objects/mug_centered_obj_normalized"
        obj_ids = [
            "1a97f3c83016abca21d0de04f408950f", "1c9f9e25c654cbca3c71bf3f4dd78475", "1eaf8db2dd2b710c7d5b1b70ae595e60",
            "3d1754b7cb46c0ce5c8081810641ef6", "4b8b10d03552e0891898dfa8eb8eefff", "4b7888feea81219ab5f4a9188bfa0ef6",
            "5c48d471200d2bf16e8a121e6886e18d", "5d72df6bc7e93e6dd0cd466c08863ebd", "5fe74baba21bba7ca4eec1b19b3a18f8",
            "6aec84952a5ffcf33f60d03e1cb068dc"]
        obj_paths = [os.path.join(base_dir, x, "models/model_normalized.obj") for x in obj_ids]
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
        rotation = None
    elif args.objects == "boxes":
        obj_paths = [f"data/boxes/train/{i}.stl" for i in range(10)]
        rotation = None
    elif args.objects == "simple_trees":
        obj_paths = [f"data/simple_trees/train/{i}.stl" for i in range(10)]
        rotation = None
    else:
        raise ValueError("Unknown object class.")

    small_objs = []
    for obj_path in obj_paths:
        small_obj = load_object_create_verts(
            obj_path, scale=None, rotation=rotation, num_surface_samples=2000, sampling_method="surface")
        small_obj = scale_object_circle(small_obj, base_scale=0.1)
        small_objs.append(small_obj)

    small_objs = [small_objs[0], small_objs[7]]
    small_obj_points = [x["points"] for x in small_objs]

    warps = warp_gen(0, small_obj_points, scale_factor=args.scale, alpha=args.alpha, visualize=args.show)
    viz_utils.show_pcds_plotly({
        "canon": small_obj_points[0],
        "orig": small_obj_points[1],
        "warped": small_obj_points[0] + warps[0].reshape(-1, 3)
    })

    dir_path = "data/warping_figure_1"
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    viz_utils.save_o3d_pcd(small_obj_points[0], os.path.join(dir_path, "canon.pcd"))
    viz_utils.save_o3d_pcd(small_obj_points[1], os.path.join(dir_path, "orig.pcd"))
    viz_utils.save_o3d_pcd(small_obj_points[0] + warps[0].reshape(-1, 3), os.path.join(dir_path, "warped.pcd"))


parser = argparse.ArgumentParser()
parser.add_argument("objects")
parser.add_argument("--scale", type=float, default=1.)
parser.add_argument("--show", default=False, action="store_true")
parser.add_argument("--n-dimensions", type=int, default=4)
parser.add_argument("--alpha", type=float, default=2.0)
main(parser.parse_args())
