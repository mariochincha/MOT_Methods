import numpy as np

from types import SimpleNamespace

from tracker.bot_sort import BoTSORT
from trackers.base_tracker import BaseTracker


class BoTSORTTracker(BaseTracker):

    def __init__(self):

        args = SimpleNamespace(

            track_high_thresh=0.3,
            track_low_thresh=0.1,
            new_track_thresh=0.3,

            track_buffer=30,
            match_thresh=0.8,

            proximity_thresh=0.5,
            appearance_thresh=0.25,

            with_reid=False,

            fast_reid_config="",
            fast_reid_weights="",

            device="cpu",

            cmc_method="orb",

            name="botsort",
            ablation=False,

            mot20=False
        )

        self.tracker = BoTSORT(args)

    def update(self, detections, frame):

        botsort_input = []

        for det in detections:

            x1, y1, x2, y2, conf, cls = det

            botsort_input.append([
                x1,
                y1,
                x2,
                y2,
                conf
            ])

        if len(botsort_input) == 0:

            botsort_input = np.empty((0, 6))

        else:

            botsort_input = np.array(
                botsort_input,
                dtype=np.float32
            )

        tracks = self.tracker.update(
            botsort_input,
            frame
        )

        print("Detecciones:", len(botsort_input))
        print("Tracks:", len(tracks))

        output_tracks = []

        for track in tracks:

            tlbr = track.tlbr

            x1 = tlbr[0]
            y1 = tlbr[1]
            x2 = tlbr[2]
            y2 = tlbr[3]

            output_tracks.append([
                x1,
                y1,
                x2,
                y2,
                track.track_id
            ])

        return np.array(output_tracks)