import itertools as it
import random

import cv2

colors: list = [c for c in it.product([0, 128, 255], repeat=3)]
colors = sorted(colors, key=lambda x: sum(x))


def detect_object(frame, req_class, precision, args):
    cfg_path: str = args.cfg
    weights_path: str = args.weights
    names_path: str = args.names
    
    net = cv2.dnn_DetectionModel(cfg_path, weights_path)
    net.setInputSize(704, 704)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    frame = cv2.resize(frame, dsize=(704, 704), interpolation=cv2.INTER_AREA)

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
        label = "%.2f" % confidence
        label = "%s" % (names[classId])
        label = "%s: %.2f" % (names[classId], confidence)
        if req_class == "anything" and confidence >= precision:
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            ) 
          
            left, top, width, height = box
            top = max(top, labelSize[1])
            cv2.rectangle(frame, box, color=final_color, thickness=3)
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
            ) 
 
            left, top, width, height = box
          
            top = max(top, labelSize[1])
            cv2.rectangle(frame, box, color=final_color, thickness=2)
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
