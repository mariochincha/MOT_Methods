import os
import glob
import cv2

from ultralytics import YOLO
from ultralytics import RTDETR

# ==========================
# CONFIGURACION
# ==========================

SEQUENCE_PATH = "VisDrone2019-MOT-val/sequences/uav0000086_00000_v"

OUTPUT_VIDEO = "detections.mp4"
OUTPUT_MOT = "detections.txt"

MODEL = YOLO("yolo26n.pt")  # O puedes usar YOLO("yolov8l.pt")


# ==========================
# LEER FRAMES
# ==========================

frame_files = sorted(
    glob.glob(
        os.path.join(SEQUENCE_PATH, "*.jpg")
    )
)

if len(frame_files) == 0:
    raise Exception("No se encontraron imágenes")


# ==========================
# CREAR VIDEO
# ==========================

first_frame = cv2.imread(frame_files[0])

height, width = first_frame.shape[:2]

video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (width, height)
)


# ==========================
# ARCHIVO MOT
# ==========================

mot_file = open(OUTPUT_MOT, "w")


# ==========================
# PROCESAMIENTO
# ==========================

for frame_id, frame_path in enumerate(frame_files, start=1):

    frame = cv2.imread(frame_path)

    results = MODEL(frame)[0]

    for det_id, box in enumerate(results.boxes, start=1):

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        conf = float(box.conf)

        cls = int(box.cls)

        class_name = MODEL.names[cls]

        w = x2 - x1
        h = y2 - y1

        # ======================
        # FORMATO MOT
        # ======================

        line = (
            f"{frame_id},"
            f"{det_id},"
            f"{x1:.2f},"
            f"{y1:.2f},"
            f"{w:.2f},"
            f"{h:.2f},"
            f"{conf:.4f},"
            f"{cls},"
            f"1\n"
        )

        mot_file.write(line)

        # ======================
        # DIBUJAR CAJAS
        # ======================

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0,255,0),
            2
        )

        label = f"{class_name} {conf:.2f}"

        cv2.putText(
            frame,
            label,
            (int(x1), int(y1)-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

    video_writer.write(frame)

    print(
        f"Frame {frame_id}/{len(frame_files)}",
        end="\r"
    )


# ==========================
# CERRAR
# ==========================

mot_file.close()

video_writer.release()

print("\nProceso terminado")