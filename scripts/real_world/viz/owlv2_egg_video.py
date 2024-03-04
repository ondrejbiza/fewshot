import requests
from PIL import Image
import torch
import cv2
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import numpy as np

device = "cpu"

video = "data/egg_cut.mp4"

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)


cap = cv2.VideoCapture(video)
i = 29
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[:, :, ::-1]
    frame = Image.fromarray(frame)
    
    # new_width = 1080
    # new_height = 1080
    # width, height = frame.size   # Get dimensions

    # left = (width - new_width)/2
    # top = (height - new_height)/2 + 400
    # right = (width + new_width)/2
    # bottom = (height + new_height)/2

    # # Crop the center of the image
    # frame = frame.crop((left, top, right, bottom))
    frame = frame.resize((frame.size[0] // 4, frame.size[1] // 4))

    # frame.show()
    print(frame.size)
    # i += 1

    # if i != 30:
    #      continue
    # else:
    #      i = 0
    #      frame.show()

    texts = [["egg"]]
    inputs = processor(text=texts, images=frame, return_tensors="pt")
    for key in inputs.keys():
        inputs[key] = inputs[key].to(device)

    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([frame.size[::-1]]).to(device)
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

    frame = np.array(frame)
    # frame = frame[:, :, ::-1]

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    for box, score, label in zip(boxes, scores, labels):
        if score.item() < 0.5:
            continue
        box = [int(round(i, 2)) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

        frame = cv2.rectangle(frame, [box[0], box[1]], [box[2], box[3]], (255,0,0), 5)
    
    cv2.imshow("image", frame)
    cv2.waitKey(1)
