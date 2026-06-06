# utils/mot_writer.py

def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter

    if union == 0:
        return 0

    return inter / union


class MOTWriter:

    def __init__(self, filename):

        self.file = open(filename, "w")

    def write(self, frame_id, tracks, detections):

        for track in tracks:

            track_id = int(track[4])

            x1, y1, x2, y2 = track[:4]

            best_cls = -1
            best_iou = 0

            for det in detections:

                dx1, dy1, dx2, dy2, conf, cls = det

                current_iou = iou(
                    [x1, y1, x2, y2],
                    [dx1, dy1, dx2, dy2]
                )

                if current_iou > best_iou:

                    best_iou = current_iou
                    best_cls = int(cls)

            # Convertir COCO -> VisDrone
            if best_cls == 0:
                best_cls = 1      # person -> pedestrian

            elif best_cls == 2:
                best_cls = 4      # car -> car

            else:
                continue          # ignorar otras clases

            w = x2 - x1
            h = y2 - y1

            line = (
                f"{frame_id},"
                f"{track_id},"
                f"{x1},{y1},{w},{h},"
                f"1,{best_cls},1\n"
            )

            self.file.write(line)

    def close(self):

        self.file.close()