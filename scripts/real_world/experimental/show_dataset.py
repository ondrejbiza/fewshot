import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import rospy

from src import viz_utils


def main(args):

    rospy.init_node("record_dataset")

    with open(args.save_file, "rb") as f:
        data = pickle.load(f)

    for i in range(len(data)):
        pcd = np.concatenate(data[i]["clouds"])
        pcd = pcd[np.random.randint(len(pcd), size=4000)]
        viz_utils.show_pcd_plotly(pcd)
        plt.subplot(3, 2, 1)
        plt.imshow(data[i]["images"][0])
        plt.subplot(3, 2, 2)
        plt.imshow(data[i]["depths"][0] / np.max(data[i]["depths"][0]))
        plt.subplot(3, 2, 3)
        plt.imshow(data[i]["images"][1])
        plt.subplot(3, 2, 4)
        plt.imshow(data[i]["depths"][1] / np.max(data[i]["depths"][1]))
        plt.subplot(3, 2, 5)
        plt.imshow(data[i]["images"][2])
        plt.subplot(3, 2, 6)
        plt.imshow(data[i]["depths"][2] / np.max(data[i]["depths"][2]))
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
