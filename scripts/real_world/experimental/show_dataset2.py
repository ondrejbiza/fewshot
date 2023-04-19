import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread

from src import viz_utils


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
    # fx = 604.364
    # fy = 603.84
    # ppx = 329.917
    # ppy = 240.609

    # Depth.
    fx = 381.814
    fy = 381.814
    ppx = 317.193
    ppy = 239.334

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

    i = 2

    pcd = data[i]["cloud"].reshape(480, 640, 3)
    image = data[i]["image"]
    depth = data[i]["depth"]
    masks = data[i]["masks"][:, 0].cpu()
    class_idx = data[i]["class_idx"]

    # I think I should do this only after I match the quantiles.
    max_distance = 1000
    depth[depth > max_distance] = max_distance

    plt.imshow(depth)
    plt.show()

    # BGR to RGB
    image = image[:, :, ::-1]

    depth2 = imread("test_depth.png")
    depth2 = depth2[:, :, 0]  # All three channels are the same.
    depth2[depth2 < 1.] = 1.
    depth2 = 1. / depth2

    plt.imshow(depth2)
    plt.show()

    depth_q = np.quantile(depth, [0.25, 0.5])
    depth2_q = np.quantile(depth2, [0.25, 0.5])

    print("depth_q", depth_q)
    print("depth2_q", depth2_q)

    depth2_n = ((depth2 - depth2_q[0]) / (depth2_q[1] - depth2_q[0])) * (depth_q[1] - depth_q[0]) + depth_q[0]

    depth2_q = np.quantile(depth2, [0.25, 0.5])
    print("depth2_q", depth2_q)

    depth2_n[depth2_n > max_distance] = max_distance

    plt.subplot(1, 2, 1)
    plt.imshow(depth)
    plt.subplot(1, 2, 2)
    plt.imshow(depth2_n)
    plt.show()

    proj, width, height = d435_intrinsics()
    pcd = depth_to_point_cloud(proj, depth)
    norm = np.sqrt(np.sum(np.square(pcd), axis=-1))
    # pcd = pcd[norm <= 0.5]
    print("@@", np.min(pcd), np.max(pcd), np.sum(np.isnan(pcd)), np.sum(np.isinf(pcd)))
    viz_utils.show_pcd_plotly(pcd, center=True)

    exit(0)

    for j in range(8):
        plt.subplot(2, 8, 1 + j)
        plt.imshow(image)
        plt.subplot(2, 8, 1 + 8 + j)
        plt.imshow(masks[j])
    plt.show()

    for j in range(len(class_idx)):
        name = f"{j}_{classes[class_idx[j]]}"
        mask = masks[j]
        tmp = pcd[mask]
        d[name] = tmp
        print(name)
        viz_utils.show_pcd_plotly(tmp, center=True)

    # viz_utils.show_pcds_plotly(d)


parser = argparse.ArgumentParser("Find objects for a particular task.")
parser.add_argument("save_file")
main(parser.parse_args())
