import argparse
from typing import Tuple
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import torch
from torchvision.models import detection
from torchvision.utils import draw_segmentation_masks
import rospy
from mask_rcnn_utils import display_instances
from online_isec.point_cloud_proxy_with_images import PointCloudProxy
import mask2former_utils

from detectron2.utils.visualizer import GenericMask


def main(args):

    # TODO: make these better
    if args.camera_index == 0:
        X_MIN = 270
        X_MAX = 720
        Y_MIN = 430
        Y_MAX = 1100
        ROT90 = False
    else:
        X_MIN = 540
        X_MAX = 1040
        Y_MIN = 30
        Y_MAX = 620
        ROT90 = True

    # setup mask2former
    predictor = mask2former_utils.get_predictor(mask2former_utils.ADE20kInstanceArgs())

    # setup camera
    rospy.init_node("mask2former_images")
    pc_proxy = PointCloudProxy()
    time.sleep(2)

    fig = plt.figure()

    while True:

        plt.clf()
        ax = fig.gca()

        _, color_image, _ = pc_proxy.get(args.camera_index)
        if color_image is None:
            print("Did not get an image.")
            time.sleep(1)
            continue

        if ROT90:
            color_image = np.rot90(color_image)
        color_image = color_image[X_MIN: X_MAX, Y_MIN: Y_MAX]
        # Mask2Former expects BGR.
        tmp = color_image[:, :, ::-1]
        predictions = predictor(tmp)
        instances = predictions["instances"].to("cpu")

        scores = instances.scores
        masks = np.asarray(instances.pred_masks)

        m = scores >= args.object_threshold
        scores = scores[m]
        masks = masks[m]

        masks = (masks > args.mask_threshold).astype(np.uint8)

        display_instances(ax, color_image, None, masks, None, None, scores)
        plt.pause(0.1)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="cpu")
parser.add_argument("-o", "--object-threshold", type=float, default=0.5)
parser.add_argument("-m", "--mask-threshold", type=float, default=0.2)
parser.add_argument("-c", "--camera-index", type=int, default=0)
main(parser.parse_args())
