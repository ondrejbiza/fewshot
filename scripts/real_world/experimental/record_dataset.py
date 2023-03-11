import argparse
import os
import pickle

import numpy as np
import rospy

from src.real_world.depth_proxy import DepthProxy
from src.real_world.image_proxy import ImageProxy
from src.real_world.point_cloud_proxy import PointCloudProxy


def main(args):

    rospy.init_node("record_dataset")
    depth_proxy = DepthProxy()
    image_proxy = ImageProxy()
    pc_proxy = PointCloudProxy()

    dir_name = os.path.dirname(args.save_file)
    if len(dir_name) > 0 and not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    data = []

    i = 0
    while True:
        input("Next")

        depths = [depth_proxy.get(i) for i in range(3)]
        images = [image_proxy.get(i) for i in range(3)]
        clouds = [pc_proxy.get(i) for i in range(3)]

        d = {
            "depths": depths,
            "images": images,
            "clouds": clouds,
        }
        data.append(d)

        with open(args.save_file, "wb") as f:
            pickle.dump(data, f)

        i += 1


parser = argparse.ArgumentParser("Find objects for a particular task.")
parser.add_argument("save_file")
main(parser.parse_args())
