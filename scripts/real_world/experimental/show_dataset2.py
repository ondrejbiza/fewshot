import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
import torch

from src import utils, viz_utils
from src.object_warping import ObjectWarpingSE2Batch, ObjectWarpingSE3Batch, warp_to_pcd_se2, warp_to_pcd_se3, PARAM_1
from src.real_world import constants


def d435_intrinsics():
    # TODO: get a real camera matrix from D435.

    width = 640
    height = 480

    # My guess.
    # fx = 608.6419
    # fy = 607.1061
    # ppx = width / 2
    # ppy = height / 2

    # Color.
    fx = 604.364
    fy = 603.84
    ppx = 329.917
    ppy = 240.609

    # Depth.
    # fx = 381.814
    # fy = 381.814
    # ppx = 317.193
    # ppy = 239.334

    proj = np.array([
        [fx, 0, ppx],
        [0, fy, ppy],
        [0, 0, 1]
    ], dtype=np.float32)
    return proj, width, height


def depth_to_point_cloud(K, D, depth_scale=1):
    height, width = D.shape

    # Invert the camera matrix
    K_inv = np.linalg.inv(K)

    # Create meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Normalize the pixel coordinates
    normalized_pixel_coords = np.stack([x, y, np.ones_like(x)], axis=-1)

    # Convert depth image to meters
    depth_meters = D * depth_scale / 1000

    # Multiply the normalized pixel coordinates by the inverse camera matrix
    camera_coords = np.matmul(normalized_pixel_coords * depth_meters[..., np.newaxis], K_inv.T)

    # Create the point cloud by reshaping the camera coordinates
    point_cloud = camera_coords.reshape(-1, 3)

    return point_cloud


def main(args):

    classes = ['cup', 'bowl', 'mug', 'bottle', 'cardboard', 'box', 'Tripod', 'Baseball bat' , 'Lamp', 'Mug Rack', 'Plate', 'Toaster', 'Spoon']

    with open(args.save_file, "rb") as f:
        data = pickle.load(f)

    with open(args.save_file2, "rb") as f:
        data2 = pickle.load(f)

    i = 1

    print(data[i].keys())
    pcd = data[i]["cloud"].reshape(480, 640, 3)
    image = data[i]["image"]
    depth = data[i]["depth"]
    masks = data[i]["masks"][:, 0].cpu().numpy()
    class_idx = data[i]["class_idx"]
    texcoords = data2[i]["texcoords"].reshape(480 * 640, 2)
    texcoords[:, 1] = np.clip(texcoords[:, 1] * 480, 0, 480 - 1)
    texcoords[:, 0] = np.clip(texcoords[:, 0] * 640, 0, 640 - 1)
    texcoords = texcoords.astype(np.int32)

    pcd = pcd.reshape(480 * 640, 3)
    # viz_utils.show_pcd_plotly(pcd, center=True)

    d = {}
    # for j in range(len(class_idx)):
    for j in [0, 2]:
        name = f"{j}_{classes[class_idx[j]]}"
        mask = masks[j]
        mask2 = mask[texcoords[:, 1], texcoords[:, 0]]
        tmp = pcd[mask2]
        tmp = tmp[np.sqrt(np.sum(np.square(tmp), axis=-1)) < 0.5]
        d[name] = tmp
        print(name)
        # viz_utils.show_pcd_plotly(tmp, center=True)

    viz_utils.show_pcds_plotly(d, center=True)

    canon_path = constants.NDF_MUGS_PCA_PATH
    canon_scale = constants.NDF_MUGS_INIT_SCALE
    canon = utils.CanonObj.from_pickle(canon_path)

    complete_pcds = {}
    for j in [0, 2]:

        name = f"{j}_{classes[class_idx[j]]}"
        mask = masks[j]
        mask2 = mask[texcoords[:, 1], texcoords[:, 0]]
        tmp = pcd[mask2]
        tmp = tmp[np.sqrt(np.sum(np.square(tmp), axis=-1)) < 0.5]

        if len(tmp) > 2000:
            tmp = utils.farthest_point_sample(tmp, 2000)[0]

        warp = ObjectWarpingSE2Batch(
            canon, tmp, torch.device("cuda:0"), **PARAM_1,
            init_scale=canon_scale)
        source_pcd_complete, _, source_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1)
        complete_pcds[j] = source_pcd_complete

    viz_utils.show_pcds_plotly(complete_pcds, center=False)


parser = argparse.ArgumentParser("Find objects for a particular task.")
parser.add_argument("save_file")
parser.add_argument("save_file2")
main(parser.parse_args())
