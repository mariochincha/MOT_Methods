# main.py

from detectors.yolo_detector import YOLODetector
from detectors.rtdetr_detector import RTDETRDetector
from detectors.yolo_sahi_detector import YOLOSAHIDetector

from trackers.sort_tracker import SORTTracker
from trackers.bytetrack_tracker import ByteTrackTracker
from trackers.botsort_tracker import BoTSORTTracker

from utils.mot_writer import MOTWriter

import os
import glob
import cv2
import time


def run_pipeline(
        detector,
        tracker,
        sequence_path,
        mot_writer):

    frame_files = sorted(
        glob.glob(
            os.path.join(sequence_path, "*.jpg")
        )
    )

    for frame_id, frame_file in enumerate(frame_files, start=1):

        frame = cv2.imread(frame_file)

        if frame is None:
            continue

        detections = detector.detect(frame)

        if isinstance(
            tracker,
            (BoTSORTTracker, ByteTrackTracker)
        ):

            tracks = tracker.update(
                detections,
                frame
            )

        else:

            tracks = tracker.update(
                detections
            )

        mot_writer.write(
            frame_id,
            tracks,
            detections
        )

    mot_writer.close()


SEQUENCE_PATH = (
    "VisDrone2019-MOT-val/sequences/uav0000086_00000_v"
)

METHODS = [

    (
        "Method 1 - YOLO11 + SORT",
        YOLODetector("yolo11n.pt"),
        SORTTracker(),
        "output_method1.txt"
    ),

    (
        "Method 2 - YOLO26 + ByteTrack",
        YOLODetector("yolo26n.pt"),
        ByteTrackTracker(),
        "output_method2.txt"
    ),

    (
        "Method 3 - RTDETR + BoTSORT",
        RTDETRDetector("rtdetr-l.pt"),
        BoTSORTTracker(),
        "output_method3.txt"
    ),

    (
        "Method 4 - YOLO11 + SAHI + SORT",
        YOLOSAHIDetector("yolo11n.pt"),
        SORTTracker(),
        "output_method4.txt"
    )

]

frame_files = sorted(
    glob.glob(
        os.path.join(SEQUENCE_PATH, "*.jpg")
    )
)

num_frames = len(frame_files)

for name, detector, tracker, output_file in METHODS:

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    writer = MOTWriter(output_file)

    start = time.time()

    run_pipeline(
        detector,
        tracker,
        SEQUENCE_PATH,
        writer
    )

    end = time.time()

    total_time = end - start

    fps = num_frames / total_time

    print(f"FPS: {fps:.2f}")
    print(f"Archivo MOT: {output_file}")