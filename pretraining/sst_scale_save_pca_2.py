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
        num_surface_samples = 10000
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
        num_surface_samples = 10000
    elif args.objects == "ndf_bottles":
        base_dir = "data/ndf_objects/bottle_centered_obj_normalized"
        obj_ids = [
            "1ae823260851f7d9ea600d1a6d9f6e07", "1b64b36bf7ddae3d7ad11050da24bb12",
            "1cf98e5b6fff5471c8724d5673a063a6", "1d4480abe9aa45ce51a99c0e19a8a54",
            "1df41477bce9915e362078f6fc3b29f5", "1e5abf0465d97d826118a17db9de8c0",
            "1ef68777bfdb7d6ba7a07ee616e34cd7", "1ffd7113492d375593202bf99dddc268",
            "2a3e0c1cd0e9076cddf5870150a75bc", "2a9817a43c5b3983bb13793251b29587"
        ]
        obj_paths = [os.path.join(base_dir, x, "models/model_normalized.obj") for x in obj_ids]
        rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()
        num_surface_samples = 10000
    elif args.objects == "ndf_trees":
        base_dir = "data/syn_racks_easy"
        obj_ids = [f"syn_rack_{i}.obj" for i in range(10)]
        obj_paths = [os.path.join(base_dir, obj_id) for obj_id in obj_ids]
        rotation = None
        num_surface_samples = 2000
    elif args.objects == "boxes":
        obj_paths = [f"data/boxes/train/{i}.stl" for i in range(10)]
        rotation = None
        num_surface_samples = 2000
    elif args.objects == "simple_trees":
        obj_paths = [f"data/simple_trees/train/{i}.stl" for i in range(10)]
        rotation = None
        num_surface_samples = 2000
    else:
        raise ValueError("Unknown object class.")

    objs = []
    small_objs = []
    for obj_path in obj_paths:
        obj = load_object_create_verts(
            obj_path, scale=None, rotation=rotation, num_surface_samples=num_surface_samples)
        small_obj = load_object_create_verts(
            obj_path, scale=None, rotation=rotation, num_surface_samples=2000, sampling_method="surface")

        obj = scale_object_circle(obj, base_scale=0.1)
        small_obj = scale_object_circle(small_obj, base_scale=0.1)

        objs.append(obj)
        small_objs.append(small_obj)

    if args.show:
        print("Large PCs:")
        tmp = []
        for i, x in enumerate(objs):
            tmp2 = np.copy(x["points"])
            tmp2[:, 0] += i * 2.
            tmp.append(tmp2)
        tmp = np.concatenate(tmp, axis=0)
        viz_utils.show_pcd_plotly(tmp, center=True)

        print("Small PCs:")
        tmp = []
        for i, x in enumerate(small_objs):
            tmp2 = np.copy(x["points"])
            tmp2[:, 0] += i * 2.
            tmp.append(tmp2)
        tmp = np.concatenate(tmp, axis=0)
        viz_utils.show_pcd_plotly(tmp, center=True)

    # We use large point clouds to figure out the canonical object.
    obj_points = [x["points"] for x in objs]
    small_obj_points = [x["points"] for x in small_objs]

    cost_sums = []
    for i in range(len(small_obj_points)):
        warps, costs = warp_gen(i, small_obj_points, scale_factor=args.scale, alpha=args.alpha, visualize=args.show)
        cost_sums.append(np.sum(costs))
    canonical_idx = np.argmin(cost_sums)

    if args.objects == "ndf_bottles":
        # Some bottles have meshes with too many points, which makes warping too slow.
        canonical_idx = 9
    print(f"Canonical obj index: {canonical_idx}.")

    if args.show:
        viz_utils.show_pcd_plotly(obj_points[canonical_idx], center=True)

    # We use small point clouds, except for the canonical object, to figure out the warps.
    tmp_obj_points = [x["points"] for x in small_objs]
    tmp_obj_points[canonical_idx] = obj_points[canonical_idx]

    warps, _ = warp_gen(canonical_idx, tmp_obj_points, scale_factor=args.scale, alpha=args.alpha, visualize=args.show)
    _, pca = pca_transform(warps, n_dimensions=args.n_dimensions)

    with open(args.save_path, "wb") as f:
        pickle.dump({
            "pca": pca,
            "canonical_obj": obj_points[canonical_idx],
            "canonical_mesh_points": objs[canonical_idx]["mesh_points"],
            "canonical_mesh_faces": objs[canonical_idx]["faces"],
            "scale": args.scale
        }, f)


parser = argparse.ArgumentParser()
parser.add_argument("save_path")
parser.add_argument("objects")
parser.add_argument("--scale", type=float, default=1.)
parser.add_argument("--show", default=False, action="store_true")
parser.add_argument("--n-dimensions", type=int, default=4)
parser.add_argument("--alpha", type=float, default=2.0)
main(parser.parse_args())
