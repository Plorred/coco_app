import cv2
from matplotlib import colors
import numpy as np
import os
import streamlit as st

COLORS = {
    u'person':(255,255,0),
    u'bicycle':(0,255,255),
    u'car':(8, 232, 200),
    u'motorcycle':(225, 158, 93),
    u'airplane':(128, 95, 253),
    u'bus':(37, 211, 222),
    u'train':(145, 53, 2),
    u'truck':(217, 41, 182),
    u'boat':(150, 200, 71),
    u'traffic light':(173, 28, 85),
    u'fire hydrant':(185, 167, 254),
    u'stop sign':(12, 191, 64),
    u'parking meter':(2, 8, 243),
    u'bench':(191, 72, 221),
    u'bird':(181, 122, 243),
    u'cat':(207, 25, 19),
    u'dog':(104, 139, 5),
    u'horse':(56, 69, 192),
    u'sheep':(30, 194, 189),
    u'cow':(51, 186, 203),
    u'elephant':(179, 32, 208),
    u'bear':(133, 227, 157),
    u'zebra':(0, 40, 192),
    u'giraffe':(253, 29, 7),
    u'backpack':(47, 33, 101),
    u'umbrella':(114, 155, 237),
    u'handbag':(75, 171, 199),
    u'tie':(32, 211, 146),
    u'suitcase':(49, 152, 42),
    u'frisbee':(212, 123, 75),
    u'skis':(212, 227, 156),
    u'snowboard':(249, 155, 8),
    u'sports ball':(118, 0, 86),
    u'kite':(4, 96, 9),
    u'baseball bat':(35, 17, 126),
    u'baseball glove':(95, 16, 189),
    u'skateboard':(41, 116, 97),
    u'surfboard':(152, 49, 203),
    u'tennis racket':(59, 117, 197),
    u'bottle':(95, 87, 166),
    u'wine glass':(36, 248, 208),
    u'cup':(186, 170, 3),
    u'fork':(136, 59, 148),
    u'knife':(22, 177, 142),
    u'spoon':(56, 244, 37),
    u'bowl':(23, 6, 130),
    u'banana':(83, 80, 14),
    u'apple':(24, 30, 214),
    u'sandwich':(211, 76, 241),
    u'orange':(63, 190, 195),
    u'broccoli':(181, 78, 220),
    u'carrot':(236, 92, 62),
    u'hot dog':(9, 234, 231),
    u'pizza':(162, 234, 137),
    u'donut':(200, 218, 5),
    u'cake':(152, 1, 190),
    u'chair':(37, 228, 16),
    u'couch':(151, 255, 165),
    u'potted plant':(79, 182, 80),
    u'bed':(48, 31, 133),
    u'dining table':(93, 83, 171),
    u'toilet':(116, 34, 102),
    u'tv':(248, 205, 108),
    u'laptop':(196, 0, 90),
    u'mouse':(78, 182, 241),
    u'remote':(255, 182, 135),
    u'keyboard':(1, 187, 204),
    u'bed':(23, 11, 232),
    u'microwave':(58, 141, 227),
    u'oven':(46, 40, 98),
    u'toaster':(37, 121, 94),
    u'sink':(76, 123, 242),
    u'refrigerator':(16, 33, 204),
    u'book':(100, 54, 17),
    u'clock':(57, 52, 119),
    u'vase':(85, 19, 224),
    u'scissors':(170, 244, 107),
    u'teddy bear':(216, 95, 147),
    u'hair drier':(89, 241, 252),
    u'toothbrush':(202, 177, 149) 
}

def detect_object(frame, req_class, precision):
    cfg_path = os.path.abspath('yolo/yolov4.cfg')
    weights_path = os.path.abspath('yolo/yolov4.weights')
    names_path = os.path.abspath('yolo/coco.names')

    # Load Yolo
    # net = cv2.dnn_DetectionModel('yolov4.cfg', 'yolov4.weights')
    net = cv2.dnn_DetectionModel(cfg_path, weights_path)
    net.setInputSize(704, 704)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    # frame = cv2.imread('sample1.jpg')
    # print(type(frame))
    # Resize the image
    frame = cv2.resize(frame, dsize=(704, 704), interpolation=cv2.INTER_AREA)

    # with open('coco.names', 'rt') as f:
    with open(names_path, 'rt') as f:
        names = f.read().rstrip('\n').split('\n')

    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    # print("Cl:", classes)
    # print("Co:", confidences)
    # print("Clf:", classes.flatten())
    # print("Cof:", confidences.flatten())
    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        # print(classId, confidence, box)
        label = '%.2f' % confidence
        label = '%s' % (names[classId])
        label = '%s: %.2f' % (names[classId], confidence)
        if names[classId] != req_class:
            continue
        elif confidence < precision:
            continue
        else:
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # fontScale: 0.5, thickness: 1
        # print(labelSize, baseLine)
            left, top, width, height = box
        # print("T:", top)
            top = max(top, labelSize[1])
        # print("MT:", top)
            cv2.rectangle(frame, box, color=COLORS[req_class], thickness=3)
        # Draw rectangle for labels
            cv2.rectangle(frame, (left - 2, top - labelSize[1] - 5), (left + labelSize[0], top - 1), COLORS[req_class], cv2.FILLED)
            cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

    return frame