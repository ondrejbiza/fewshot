import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import torch
from torchvision.models import detection
from torchvision.utils import draw_segmentation_masks
import rospy
from mask_rcnn_utils import display_instances, coco_class_names
from online_isec.point_cloud_proxy_with_images import PointCloudProxy

# TODO: make these better
X_MIN = 540
X_MAX = 1040
Y_MIN = 30
Y_MAX = 620
ROT90 = True

X_MIN = 270
X_MAX = 720
Y_MIN = 430
Y_MAX = 1100
ROT90 = False


def main(args):

    # setup mask r-cnn
    torch_device = torch.device("cuda:0")
    model = detection.maskrcnn_resnet50_fpn(pretrained=True).to(torch_device)
    model.eval()

    # setup camera
    rospy.init_node("mask_rcnn_images")
    pc_proxy = PointCloudProxy()
    time.sleep(2)

    fig = plt.figure()

    while True:

        plt.clf()
        ax = fig.gca()

        _, color_image, _ = pc_proxy.get(0)
        if color_image is None:
            print("Did not get an image.")
            time.sleep(1)
            continue

        if ROT90:
            color_image = np.rot90(color_image)
        color_image = color_image[X_MIN: X_MAX, Y_MIN: Y_MAX]
        image_pt = torch.tensor(color_image / 255., dtype=torch.float32, device=torch_device)[None].permute((0, 3, 1, 2))

        with torch.no_grad():
            segm_d = model(image_pt)[0]

        for key, value in segm_d.items():
            segm_d[key] = value.cpu()

        if len(segm_d["labels"]) == 0:
            continue

        boxes = segm_d["boxes"].numpy()
        masks = segm_d["masks"][:, 0].numpy()
        labels = segm_d["labels"].numpy()
        scores = segm_d["scores"].numpy()

        m = scores > args.object_threshold

        boxes = boxes[m]
        masks = masks[m]
        labels = labels[m]
        scores = scores[m]

        masks = (masks > args.mask_threshold).astype(np.uint8)

        display_instances(ax, color_image, boxes, masks, labels, coco_class_names, scores)
        plt.pause(0.1)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="cpu")
parser.add_argument("-o", "--object-threshold", type=float, default=0.5)
parser.add_argument("-m", "--mask-threshold", type=float, default=0.2)
main(parser.parse_args())
