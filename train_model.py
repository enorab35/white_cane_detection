# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:58:03 2024

@author: maelb
"""

from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

results = model.train(data='dataset.yaml', epochs=3)
