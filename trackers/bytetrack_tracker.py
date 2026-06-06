# trackers/bytetrack_tracker.py

import numpy as np
from types import SimpleNamespace

from tracker_byte.byte_tracker import BYTETracker
from trackers.base_tracker import BaseTracker


class ByteTrackTracker(BaseTracker):

    def __init__(self):

        args = SimpleNamespace(

            track_thresh=0.3,
            track_buffer=100,
            match_thresh=0.9,

            mot20=False
        )

        self.tracker = BYTETracker(args)

    def update(self, detections, frame):

        byte_input = []

        for det in detections:

            x1, y1, x2, y2, conf, cls = det

            byte_input.append([
                x1,
                y1,
                x2,
                y2,
                conf
            ])

        if len(byte_input) == 0:

            byte_input = np.empty((0, 5))

        else:

            byte_input = np.array(
                byte_input,
                dtype=np.float32
            )

        h, w = frame.shape[:2]

        tracks = self.tracker.update(
            byte_input,
            (h, w),
            (h, w)
        )

        output_tracks = []

        for track in tracks:

            tlbr = track.tlbr

            output_tracks.append([
                tlbr[0],
                tlbr[1],
                tlbr[2],
                tlbr[3],
                track.track_id
            ])

        return np.array(output_tracks)