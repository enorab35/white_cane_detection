# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:58:03 2024

@author: maelb
"""

from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

results = model.train(data='training/dataset.yaml', epochs=100)
model = YOLO("runs/detect/train/weights/best.pt")
results = model.val(data="test/dataset.yaml")
