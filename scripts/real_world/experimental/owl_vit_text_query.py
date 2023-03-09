import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import torch
from torchvision.models import detection
from torchvision.utils import draw_segmentation_masks
import rospy
from src.real_world.image_proxy import ImageProxy
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
    # text_queries = ["mug", "cup", "measuring cup", "mug tree", "top of a mug", "mug from the top", "mug from top", "glass"]
    text_queries = ["mug rack"]

    # setup camera
    rospy.init_node("mask_rcnn_images")
    image_proxy = ImageProxy()
    time.sleep(2)

    fig = plt.figure()

    while True:

        plt.clf()
        ax = fig.gca()

        color_image = image_proxy.get(args.camera_index)
        if color_image is None:
            print("Did not get an image.")
            time.sleep(1)
            continue

        # if ROT90:
            # color_image = np.rot90(color_image)
        # color_image = color_image[X_MIN: X_MAX, Y_MIN: Y_MAX]
        # color_image = color_image[CENTER[0] - SIZE: CENTER[0] + SIZE, CENTER[1] - SIZE: CENTER[1] + SIZE]
        image = Image.fromarray(color_image.astype(np.uint8)).convert("RGB")

        # plt.imshow(image)
        # plt.show()

        inputs = processor(text=text_queries, images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        for k, val in outputs.items():
            if k not in {"text_model_output", "vision_model_output"}:
                print(f"{k}: shape of {val.shape}")

        print("\nText model outputs")
        for k, val in outputs.text_model_output.items():
            print(f"{k}: shape of {val.shape}")

        print("\nVision model outputs")
        for k, val in outputs.vision_model_output.items():
            print(f"{k}: shape of {val.shape}") 

        mixin = ImageFeatureExtractionMixin()

        # Load example image
        image_size = model.config.vision_config.image_size
        image = mixin.resize(image, image_size)
        input_image = np.asarray(image).astype(np.float32) / 255.0

        # Threshold to eliminate low probability predictions
        score_threshold = 0.1

        # Get prediction logits
        logits = torch.max(outputs["logits"][0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().detach().numpy()

        # Get prediction labels and boundary boxes
        labels = logits.indices.cpu().detach().numpy()
        boxes = outputs["pred_boxes"][0].cpu().detach().numpy()

        def plot_predictions(input_image, text_queries, scores, boxes, labels):
            ax.imshow(input_image, extent=(0, 1, 1, 0))
            ax.set_axis_off()

            for score, box, label in zip(scores, boxes, labels):
                if score < score_threshold:
                    continue

                cx, cy, w, h = box
                ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
                        [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
                ax.text(
                    cx - w / 2,
                    cy + h / 2 + 0.015,
                    f"{text_queries[label]}: {score:1.2f}",
                    ha="left",
                    va="top",
                    color="red",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "red",
                        "boxstyle": "square,pad=.3"
                    })
        
        plot_predictions(input_image, text_queries, scores, boxes, labels)
        plt.pause(0.1)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", default="cpu")
parser.add_argument("-o", "--object-threshold", type=float, default=0.5)
parser.add_argument("-m", "--mask-threshold", type=float, default=0.2)
parser.add_argument("-c", "--camera-index", type=int, default=0)
main(parser.parse_args())
