import cv2
import glob
import os

SEQUENCE_PATH = "VisDrone2019-MOT-val/sequences/uav0000086_00000_v"

ANNOTATION_FILE = \
"output_method1.txt"

OUTPUT_VIDEO = "result_method1.mp4"

annotations = {}

with open(ANNOTATION_FILE, "r") as f:

    for line in f:

        fields = line.strip().split(",")

        frame_id = int(fields[0])

        track_id = int(fields[1])

        x = float(fields[2])
        y = float(fields[3])

        w = float(fields[4])
        h = float(fields[5])

        cls = int(fields[7])

        if frame_id not in annotations:
            annotations[frame_id] = []

        annotations[frame_id].append(
            (track_id, x, y, w, h, cls)
        )

frame_files = sorted(
    glob.glob(
        os.path.join(SEQUENCE_PATH, "*.jpg")
    )
)

first = cv2.imread(frame_files[0])

height, width = first.shape[:2]

video = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (width, height)
)

for frame_id, frame_path in enumerate(frame_files, start=1):

    frame = cv2.imread(frame_path)

    objects = annotations.get(frame_id, [])

    for track_id, x, y, w, h, cls in objects:

        x1 = int(x)
        y1 = int(y)

        x2 = int(x + w)
        y2 = int(y + h)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        label = f"ID:{track_id} CLS:{cls}"

        cv2.putText(
            frame,
            label,
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

    video.write(frame)

video.release()

count = 0

with open("output_method1.txt") as f:
    for line in f:
        if line.startswith("1,"):
            count += 1

print(count)