# evaluator.py

import motmetrics as mm
import pandas as pd
import time

from torch import gt


class MOTEvaluator:

    def __init__(self, gt_file, pred_file):

        self.gt_file = gt_file
        self.pred_file = pred_file

    def load_mot(self, filename):

        cols = [
            "frame",
            "id",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "class",
            "truncation",
            "occlusion"
        ]

        df = pd.read_csv(
            filename,
            header=None,
            names=cols
        )

        return df

    def evaluate(self):

        gt = self.load_mot(self.gt_file)
        pred = self.load_mot(self.pred_file)

        gt = gt[
            gt["class"].isin([1, 4])
        ]

        pred = pred[
            pred["class"].isin([1, 4])

        ]

        print("GT filtrado:")
        print(gt["class"].value_counts())

        print("Pred filtrado:")
        print(pred["class"].value_counts())

        acc = mm.MOTAccumulator(auto_id=True)

        frames = sorted(
            set(gt["frame"].unique())
            | set(pred["frame"].unique())
        )

        for frame in frames:

            gt_frame = gt[
                gt["frame"] == frame
            ]

            pred_frame = pred[
                pred["frame"] == frame
            ]

            gt_ids = gt_frame["id"].tolist()

            pred_ids = pred_frame["id"].tolist()

            gt_boxes = gt_frame[
                ["x", "y", "w", "h"]
            ].values

            pred_boxes = pred_frame[
                ["x", "y", "w", "h"]
            ].values

            distances = mm.distances.iou_matrix(
                gt_boxes,
                pred_boxes,
                max_iou=0.5
            )

            acc.update(
                gt_ids,
                pred_ids,
                distances
            )

        mh = mm.metrics.create()

        summary = mh.compute(
            acc,
            metrics=[
                "mota",
                "idf1",
                "num_switches"
            ],
            name="tracking"
        )

        print("\n========== RESULTADOS ==========\n")

        print(
            summary[
                ["mota", "idf1", "num_switches"]
            ]
        )

        return summary