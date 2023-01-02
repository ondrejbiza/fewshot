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
from online_isec.point_cloud_proxy import PointCloudProxy


def main(args):

    # setup mask r-cnn
    torch_device = torch.device("cuda:0")
    model = detection.maskrcnn_resnet50_fpn(pretrained=True).to(torch_device)
    model.eval()

    # setup camera
    rospy.init_node("test")
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

        color_image = np.rot90(color_image)
        color_image = color_image[540: 1040, 30: 620]
        image_pt = torch.tensor(color_image / 255., dtype=torch.float32, device=torch_device)[None].permute((0, 3, 1, 2))

        with torch.no_grad():
            segm_d = model(image_pt)[0]

        for key, value in segm_d.items():
            segm_d[key] = value.cpu()

        if len(segm_d["labels"]) == 0:
            continue

        # I would need to filter by instances here. Doesn't display class labels.
        # segm = segm_d["masks"][:, 0]
        # images = draw_segmentation_masks((image_pt[0].cpu() * 255).type(torch.uint8), segm.cpu() > 0.5)
        # images = images.permute((1, 2, 0))
        # images = images.cpu().numpy()

        boxes = segm_d["boxes"].numpy()
        masks = segm_d["masks"][:, 0].numpy()
        labels = segm_d["labels"].numpy()
        scores = segm_d["scores"].numpy()

        m = scores > args.object_threshold

        boxes = boxes[m]
        masks = masks[m]
        labels = labels[m]
        scores = scores[m]

        masks = masks.transpose((1, 2, 0))
        masks = (masks > args.mask_threshold).astype(np.int32)

        display_instances(color_image, boxes, masks, labels, coco_class_names, scores=scores, show_bbox=False, ax=ax)
        plt.pause(0.1)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="cpu")
parser.add_argument("-o", "--object-threshold", type=float, default=0.5)
parser.add_argument("-m", "--mask-threshold", type=float, default=0.2)
main(parser.parse_args())
