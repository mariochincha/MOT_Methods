# trackers/sort_tracker.py

import numpy as np

from utils.sort import Sort
from trackers.base_tracker import BaseTracker


class SORTTracker(BaseTracker):

    def __init__(self):

        self.tracker = Sort()

    def update(self, detections):

        sort_input = []

        for det in detections:

            x1,y1,x2,y2,conf,cls = det

            sort_input.append(
                [x1,y1,x2,y2,conf]
            )

        if len(sort_input) == 0:

            sort_input = np.empty((0,5))

        else:

            sort_input = np.array(
                sort_input,
                dtype=np.float32
            )

        return self.tracker.update(
            sort_input
        )