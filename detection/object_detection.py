import itertools as it
import os
import random

import cv2
import numpy as np
import streamlit as st
from matplotlib import colors

colors: list = [c for c in it.product([0, 128, 255], repeat=3)]
colors = sorted(colors, key=lambda x: sum(x))


def detect_object(frame, req_class, precision):
    cfg_path: str = os.path.abspath("yolo/yolov4.cfg")
    weights_path: str = os.path.abspath("yolo/yolov4.weights")
    names_path: str = os.path.abspath("yolo/coco.names")

    # Load Yolo
    # net = cv2.dnn_DetectionModel('yolov4.cfg', 'yolov4.weights')
    net = cv2.dnn_DetectionModel(cfg_path, weights_path)
    net.setInputSize(704, 704)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    # Resize the image
    frame = cv2.resize(frame, dsize=(704, 704), interpolation=cv2.INTER_AREA)

    # with open('coco.names', 'rt') as f:
    with open(names_path, "rt") as f:
        names: list = f.read().rstrip("\n").split("\n")

    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    final_color = random.choice(colors)
    text_color = (0, 0, 0)
    if final_color == (0, 0, 0):
        text_color = (255, 255, 255)
    for classId, confidence, box in zip(
        classes.flatten(), confidences.flatten(), boxes
    ):
        # print(classId, confidence, box)
        label = "%.2f" % confidence
        label = "%s" % (names[classId])
        label = "%s: %.2f" % (names[classId], confidence)
        if req_class == "anything" and confidence >= precision:
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )  # fontScale: 0.5, thickness: 1
            # print(labelSize, baseLine)
            left, top, width, height = box
            # print("T:", top)
            top = max(top, labelSize[1])
            # print("MT:", top)
            cv2.rectangle(frame, box, color=final_color, thickness=3)
            # Draw rectangle for labels
            cv2.rectangle(
                frame,
                (left - 2, top - labelSize[1] - 5),
                (left + labelSize[0], top - 1),
                final_color,
                cv2.FILLED,
            )
            cv2.putText(
                frame, label, (left, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color
            )
        if names[classId] != req_class:
            continue
        elif confidence < precision:
            continue
        else:
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )  # fontScale: 0.5, thickness: 1
            # print(labelSize, baseLine)
            left, top, width, height = box
            # print("T:", top)
            top = max(top, labelSize[1])
            # print("MT:", top)
            cv2.rectangle(frame, box, color=final_color, thickness=2)
            # Draw rectangle for labels
            cv2.rectangle(
                frame,
                (left - 2, top - labelSize[1] - 5),
                (left + labelSize[0], top - 1),
                final_color,
                cv2.FILLED,
            )
            cv2.putText(
                frame, label, (left, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color
            )

    return frame
