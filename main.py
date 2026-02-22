import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

video_choice = input("Enter traffic level (low / medium / high): ")
cap = cv2.VideoCapture(f"videos/{video_choice}.mp4")

vehicle_classes = [2, 3, 5, 7]
frame_counts = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    # Count vehicles
    count = 0
    for box in results[0].boxes:
        if int(box.cls) in vehicle_classes:
            count += 1

    frame_counts.append(count)
    if len(frame_counts) > 150:
        frame_counts.pop(0)

    avg_count = int(np.mean(frame_counts))

    # Density → Green Time
    if avg_count <= 9:
        density = "LOW"
        green_time = 15
    elif avg_count <= 18:
        density = "MEDIUM"
        green_time = 30
    else:
        density = "HIGH"
        green_time = 45

    # Dashboard background
    cv2.rectangle(annotated_frame, (10, 10), (450, 160), (0, 0, 0), -1)

    # Display info
    cv2.putText(annotated_frame, f"Vehicles: {avg_count}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(annotated_frame, f"Density: {density}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(annotated_frame, f"Green Signal Time: {green_time} sec", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("AI Traffic Management System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()