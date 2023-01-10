import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import open3d as o3d
from sst_utils import load_object_create_verts, pick_canonical, cpd_transform, cpd_transform_plot, warp_gen, \
    pca_transform, pca_reconstruct


def main():

    example_pc = np.load("data/example_mug_pc.npy")
    example_pc -= np.mean(example_pc, axis=0, keepdims=True)

    base_dir = "data/ndf_objects/mug_centered_obj_normalized"
    # obj_ids = ["1a97f3c83016abca21d0de04f408950f", "1ae1ba5dfb2a085247df6165146d5bbd", "1bc5d303ff4d6e7e1113901b72a68e7c",
    #            "1be6b2c84cdab826c043c2d07bb83fc8", "1c3fccb84f1eeb97a3d0a41d6c77ec7c", "1c9f9e25c654cbca3c71bf3f4dd78475",
    #            "1d18255a04d22794e521eeb8bb14c5b3", "1ea9ea99ac8ed233bf355ac8109b9988", "1eaf8db2dd2b710c7d5b1b70ae595e60",
    #            "1f035aa5fc6da0983ecac81e09b15ea9"]
    obj_ids = [
        "1a97f3c83016abca21d0de04f408950f", "1c9f9e25c654cbca3c71bf3f4dd78475", "1eaf8db2dd2b710c7d5b1b70ae595e60",
        "3d1754b7cb46c0ce5c8081810641ef6", "4b8b10d03552e0891898dfa8eb8eefff", "4b7888feea81219ab5f4a9188bfa0ef6",
        "5c48d471200d2bf16e8a121e6886e18d", "5d72df6bc7e93e6dd0cd466c08863ebd", "5fe74baba21bba7ca4eec1b19b3a18f8",
        "6aec84952a5ffcf33f60d03e1cb068dc"]
    obj_paths = [os.path.join(base_dir, x, "models/model_normalized.obj") for x in obj_ids]
    scale = 0.2
    rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()
    voxel_size = 0.0075  # usually 2k to 3k

    objs = []
    for obj_path in obj_paths:
        obj = load_object_create_verts(
            obj_path, voxel_size=voxel_size, scale=scale, rotation=rotation)
        objs.append(obj)
    obj_points = [x["points"] for x in objs]

    pc = example_pc
    for i in range(len(obj_points)):
        tmp = np.copy(obj_points[i])
        tmp[..., 0] += (i + 1) * 0.2
        pc = np.concatenate([pc, tmp])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw([pcd])


main()
