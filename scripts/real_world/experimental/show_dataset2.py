import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread

from src import viz_utils


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

    # BGR to RGB
    image = image[:, :, ::-1]

    depth2 = imread("test_depth.png")

    print(np.min(depth), np.max(depth), np.mean(depth))
    print(np.min(pcd), np.max(pcd), np.mean(pcd))
    print(np.min(depth2), np.max(depth2), np.mean(depth2))

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
