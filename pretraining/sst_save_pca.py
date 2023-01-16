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

    np.random.seed(2023)
    trimesh.util.attach_to_log()

    if args.objects == "ndf_mugs":
        base_dir = "data/ndf_objects/mug_centered_obj_normalized"
        # Some (most?) of these mugs are closed. I can't used them for motion planning.
        # obj_ids = ["1a97f3c83016abca21d0de04f408950f", "1ae1ba5dfb2a085247df6165146d5bbd", "1bc5d303ff4d6e7e1113901b72a68e7c",
        #     "1be6b2c84cdab826c043c2d07bb83fc8", "1c3fccb84f1eeb97a3d0a41d6c77ec7c", "1c9f9e25c654cbca3c71bf3f4dd78475",
        #     "1d18255a04d22794e521eeb8bb14c5b3", "1ea9ea99ac8ed233bf355ac8109b9988", "1eaf8db2dd2b710c7d5b1b70ae595e60",
        #     "1f035aa5fc6da0983ecac81e09b15ea9"]
        obj_ids = [
            "1a97f3c83016abca21d0de04f408950f", "1c9f9e25c654cbca3c71bf3f4dd78475", "1eaf8db2dd2b710c7d5b1b70ae595e60",
            "3d1754b7cb46c0ce5c8081810641ef6", "4b8b10d03552e0891898dfa8eb8eefff", "4b7888feea81219ab5f4a9188bfa0ef6",
            "5c48d471200d2bf16e8a121e6886e18d", "5d72df6bc7e93e6dd0cd466c08863ebd", "5fe74baba21bba7ca4eec1b19b3a18f8",
            "6aec84952a5ffcf33f60d03e1cb068dc"]
        obj_paths = [os.path.join(base_dir, x, "models/model_normalized.obj") for x in obj_ids]
        scale = 0.14
        rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()
        voxel_size = 0.0075  # usually 2k to 3k
    else:
        raise ValueError("Unknown object class.")

    objs = []
    for obj_path in obj_paths:
        if obj_path[-3:] == "npy":
            objs.append(np.load(obj_path))
            # TODO: rotation and scaling
        else:
            obj = load_object_create_verts(
                obj_path, voxel_size=voxel_size, scale=scale, rotation=rotation)
            objs.append(obj)

    obj_points = [x["points"] for x in objs]

    tmp = np.concatenate(obj_points, axis=0)
    print("PC stats:")
    print("x min {:f} x max {:f} x mean {:f}".format(tmp[:, 0].min(), tmp[:, 0].max(), tmp[:, 0].mean()))
    print("y min {:f} y max {:f} y mean {:f}".format(tmp[:, 1].min(), tmp[:, 1].max(), tmp[:, 1].mean()))
    print("z min {:f} z max {:f} z mean {:f}".format(tmp[:, 2].min(), tmp[:, 2].max(), tmp[:, 2].mean()))

    print("Picking canonical object.")
    canonical_idx = pick_canonical(obj_points)
    print("Canonical obj index: {:d}.".format(canonical_idx))

    if args.show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, obj in enumerate(obj_points):
            if i == canonical_idx:
                ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], color=[1., 0., 0., 1.])
            else:
                ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], color=[0.9, 0.9, 0.9, 0.5])
        plt.show()

    warps = warp_gen(canonical_idx, obj_points, scale_factor=args.scale, visualize=args.show)
    components, pca = pca_transform(warps, n_dimensions=args.n_dimensions)

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
main(parser.parse_args())
