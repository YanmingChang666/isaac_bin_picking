from ultralytics import YOLO
import cv2
import numpy as np
# Load a model
model = YOLO("best.pt")

# Predict with the model
results = model("./data_for_test/img14.jpg", save=True, conf=0.8)  # predict on an image

i = 0
for r in results:
    masks = r.masks

    for mask in masks:
        x = mask.data.to('cpu').detach().numpy().copy()
        bolt_mask = x.reshape(480, 640, 1)
        bolt_mask_rgb = cv2.cvtColor(bolt_mask*255, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(f"result{i}.png", bolt_mask_rgb)
        i += 1
    
    boxes = r.boxes
    for box in boxes:
        conf = box.conf.to('cpu').detach().numpy().copy()
        print(f"conf:{conf[0]}")