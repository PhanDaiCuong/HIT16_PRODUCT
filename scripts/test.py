import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from ultralytics import solutions
if __name__ == "__main__":
    # Updated paths to be relative and cross-platform
    img_path = "D:\HIT16_PRODUCT\pklot dataset.v1i.yolov11\test\images\2012-09-12_15_15_01_jpg.rf.89f871494503919d10782b916e2394b2.jpg"  # Update with actual test image path
    label_path = "./labels/label_convert.json"  # Update with actual label path
    video_path = "D:\HIT16_PRODUCT\The Dance of a Parking Lot.webm"  # Update with actual video path
    model_path = "./models/best.pt"

    parkingmanager = solutions.ParkingManagement(
        model=model_path,
        json_file="./polygons.json",  # Now uses local polygons.json
        show_labels=False,
    )

    cap = cv2.VideoCapture(video_path)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        cnt += 1
        # Fixed: Changed bitwise OR (|) to logical OR (or)
        if cnt % 2 == 0 or cnt % 3 == 0 or cnt % 5 == 0:
            continue
        if not ret:
            break

        result = parkingmanager.process(frame).plot_im
        cv2.imshow("Parking Manager", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()