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
from mask_rcnn_utils import display_instances_masks_only
from online_isec.point_cloud_proxy import PointCloudProxy

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from prereq.Mask2Former.mask2former import add_maskformer2_config
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import GenericMask


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


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
    torch_device = torch.device("cuda:0")

    # @dataclass
    # class Args:
    #     config_file: str = "prereq/Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
    #     opts: Tuple[str, ...] = ("MODEL.WEIGHTS", "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl")
    #     confidence_threshold: float = 0.5

    # @dataclass
    # class Args:
    #     config_file: str = "prereq/Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
    #     opts: Tuple[str, ...] = ("MODEL.WEIGHTS", "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl")
    #     confidence_threshold: float = 0.5

    @dataclass
    class Args:
        config_file: str = "prereq/Mask2Former/configs/ade20k/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml"
        opts: Tuple[str, ...] = ("MODEL.WEIGHTS", "https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/instance/maskformer2_swin_large_IN21k_384_bs16_160k/model_final_92dae9.pkl")
        confidence_threshold: float = 0.5

    args2 = Args()
    cfg = setup_cfg(args2)
    predictor = DefaultPredictor(cfg)

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
        color_image = color_image[:, :, ::-1]
        print(color_image.min(), color_image.max())
        predictions = predictor(color_image)
        instances = predictions["instances"].to("cpu")

        scores = instances.scores
        masks = np.asarray(instances.pred_masks)

        m = scores >= args.object_threshold
        masks = masks[m]

        masks = [GenericMask(x, masks.shape[1], masks.shape[2]) for x in masks]
        areas = np.asarray([x.area() for x in masks])
        sorted_idxs = np.argsort(-areas).tolist()
        masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None

        display_instances_masks_only(color_image[:, :, ::-1], masks, show_bbox=False, ax=ax)
        plt.pause(0.1)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="cpu")
parser.add_argument("-o", "--object-threshold", type=float, default=0.5)
parser.add_argument("-m", "--mask-threshold", type=float, default=0.2)
parser.add_argument("-c", "--camera-index", type=int, default=0)
main(parser.parse_args())
