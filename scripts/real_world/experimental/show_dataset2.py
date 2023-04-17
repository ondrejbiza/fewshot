import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt

from src import viz_utils


def main(args):

    with open(args.save_file, "rb") as f:
        data = pickle.load(f)

    for i in range(len(data)):
        pcd = np.concatenate(data[i]["cloud"])
        pcd = pcd.reshape(-1, 3)
        pcd = pcd[np.random.randint(len(pcd), size=4000)]
        # viz_utils.show_pcd_plotly(pcd)
        plt.subplot(3, 2, 1)
        plt.imshow(data[i]["image"])
        plt.subplot(3, 2, 2)
        plt.imshow(data[i]["depth"] / np.max(data[i]["depth"]))
        plt.subplot(3, 2, 3)
        plt.imshow(data[i]["masks"][1, 0].cpu())
        plt.show()

        # Showing ordered point cloud.
        # pcd = data[i]["clouds"][0]
        # pcd = pcd.reshape(720, 1280, 3)
        # plt.subplot(1, 2, 1)
        # plt.imshow(data[i]["images"][0])
        # plt.subplot(1, 2, 2)
        # plt.imshow(pcd[:, :, 0])
        # plt.show()


parser = argparse.ArgumentParser("Find objects for a particular task.")
parser.add_argument("save_file")
main(parser.parse_args())
