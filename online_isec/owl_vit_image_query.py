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
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers.image_utils import ImageFeatureExtractionMixin
from PIL import Image


def main(args):

    if args.camera_index == 0:
        CENTER = [490 - 120, 700]
        SIZE = 230 + 120
        ROT90 = False
    else:
        X_MIN = 540
        X_MAX = 1040
        Y_MIN = 30
        Y_MAX = 620
        ROT90 = True

    # setup mask r-cnn
    # checkpoint_name = "google/owlvit-base-patch32"
    checkpoint_name = "google/owlvit-large-patch14"
    torch_device = torch.device("cuda:0")

    model = OwlViTForObjectDetection.from_pretrained(checkpoint_name).to(torch_device)
    model.eval()
    processor = OwlViTProcessor.from_pretrained(checkpoint_name)
    # plt.imshow(query_image)
    # plt.show()

    # setup camera
    rospy.init_node("mask_rcnn_images")
    pc_proxy = PointCloudProxy()
    time.sleep(2)

    fig = plt.figure()

    while True:

        query_image = Image.open("/home/ur5/Pictures/template2.webp").convert("RGB")

        plt.clf()
        ax = fig.gca()

        _, color_image, _ = pc_proxy.get(args.camera_index)
        if color_image is None:
            print("Did not get an image.")
            time.sleep(1)
            continue

        if ROT90:
            color_image = np.rot90(color_image)
        # color_image = color_image[X_MIN: X_MAX, Y_MIN: Y_MAX]
        color_image = color_image[CENTER[0] - SIZE: CENTER[0] + SIZE, CENTER[1] - SIZE: CENTER[1] + SIZE]
        image = Image.fromarray(color_image.astype(np.uint8)).convert("RGB")
        target_sizes = torch.Tensor([image.size[::-1]])

        # plt.imshow(image)
        # plt.show()

        inputs = processor(query_images=query_image, images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs)

        for k, val in outputs.items():
            if k not in {"text_model_output", "vision_model_output"}:
                print(f"{k}: shape of {val.shape}")

        print("\nVision model outputs")
        for k, val in outputs.vision_model_output.items():
            print(f"{k}: shape of {val.shape}")

        img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        outputs.logits = outputs.logits.cpu()
        outputs.target_pred_boxes = outputs.target_pred_boxes.cpu() 

        results = processor.post_process_image_guided_detection(outputs=outputs, threshold=0.99, nms_threshold=0.3, target_sizes=target_sizes)
        boxes, scores = results[0]["boxes"], results[0]["scores"]

        # Draw predicted bounding boxes
        for box, score in zip(boxes, scores):
            box = [int(i) for i in box.tolist()]

            img = cv2.rectangle(img, box[:2], box[2:], (255,0,0), 5)
            if box[3] + 25 > 768:
                y = box[3] - 10
            else:
                y = box[3] + 25 
                
        plt.imshow(img[:,:,::-1])
        plt.pause(0.1)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="cpu")
parser.add_argument("-o", "--object-threshold", type=float, default=0.5)
parser.add_argument("-m", "--mask-threshold", type=float, default=0.2)
parser.add_argument("-c", "--camera-index", type=int, default=0)
main(parser.parse_args())
