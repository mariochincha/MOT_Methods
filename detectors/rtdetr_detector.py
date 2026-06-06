# detectors/rtdetr_detector.py

from ultralytics import RTDETR
from detectors.base_detector import BaseDetector

class RTDETRDetector(BaseDetector):

    def __init__(self, weights):

        self.model = RTDETR(weights)

    def detect(self, frame):

        results = self.model(frame, classes=[0,2])[0]

        detections = []

        for box in results.boxes:

            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()

            conf = float(box.conf)

            cls = int(box.cls)

            detections.append(
                [x1,y1,x2,y2,conf,cls]
            )

        return detections