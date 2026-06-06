from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from detectors.base_detector import BaseDetector


class RTDETRSAHIDetector(BaseDetector):

    def __init__(self, model_path):

        self.model = AutoDetectionModel.from_pretrained(

            model_type="ultralytics",

            model_path=model_path,

            confidence_threshold=0.25,

            device="cuda:0"
        )

    def detect(self, frame):

        result = get_sliced_prediction(

            frame,

            self.model,

            slice_height=512,
            slice_width=512,

            overlap_height_ratio=0.25,
            overlap_width_ratio=0.25
        )

        detections = []

        for obj in result.object_prediction_list:

            bbox = obj.bbox

            detections.append([

                bbox.minx,
                bbox.miny,
                bbox.maxx,
                bbox.maxy,

                obj.score.value,

                obj.category.id
            ])

        return detections