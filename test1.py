#test to convert image sequence into a video .mp4
import cv2
import glob
import os

sequence_path = "VisDrone2019-MOT-val/sequences/uav0000086_00000_v"

frame_files = sorted(
    glob.glob(os.path.join(sequence_path, "*.jpg"))
)

# Leer primer frame para obtener tamaño
first_frame = cv2.imread(frame_files[0])

height, width = first_frame.shape[:2]

video = cv2.VideoWriter(
    "output.mkv",
    cv2.VideoWriter_fourcc(*'H264'),
    24,  # FPS
    (width, height)
)

for frame_file in frame_files:

    frame = cv2.imread(frame_file)

    video.write(frame)

video.release()

print("Video creado")