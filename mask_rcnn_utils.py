# https://raw.githubusercontent.com/matterport/Mask_RCNN/master/mrcnn/visualize.py
from typing import Optional, List
import random
import colorsys
import numpy as np
from numpy.typing import NDArray
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import patches,  lines


coco_class_names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "hair brush",
]
coco_class_names = {k + 1: v for k, v in enumerate(coco_class_names)}


def random_colors(N: int, bright: bool=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    #random.shuffle(colors)
    return colors


def display_instances(ax, image: NDArray, boxes: Optional[NDArray]=None, masks: Optional[NDArray]=None,
                        class_ids: Optional[NDArray]=None, class_names: Optional[List[str]]=None, scores: Optional[NDArray]=None):

    assert boxes is not None or masks is not None

    num_instances = boxes.shape[0] if boxes is not None else masks.shape[0]
    height, width = image.shape[:2]

    colors = random_colors(num_instances)

    ax.set_xlim(-10, width + 10)
    ax.set_ylim(height + 10, -10)
    ax.axis("off")

    for i in range(num_instances):

        if boxes is not None and np.any(boxes[i]):
            # Add bounding boxes.
            x1, y1, x2, y2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=colors[i], facecolor='none')
            ax.add_patch(p)

            if class_ids is not None and class_names is not None:
                # Add annotations with class id and score.
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
                ax.text(x1, y1 + 8, caption,
                    color="b", size=11, backgroundcolor="w")
            elif scores is not None:
                caption = "{:.3f}".format(scores[i])
                ax.text(x1, y1 + 8, caption,
                    color="b", size=11, backgroundcolor="w")

        if masks is not None:
            # Draw mask polygons.
            mask = masks[i]
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor=colors[i] + (0.3,), edgecolor=colors[i] + (0.7,))
                ax.add_patch(p)

    ax.imshow(image)
