from ultralytics import YOLO
import cv2
import numpy as np
import os

model_path = r"C:\Users\DELL\Downloads\yolov8-roadpothole-detection-main\yolov8-roadpothole-detection-main\best.pt"
video_path = r"C:\Users\DELL\Downloads\yolov8-roadpothole-detection-main\yolov8-roadpothole-detection-main\p.mp4"


if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    exit()
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found!")
    exit()


print("Loading model...")
model = YOLO(model_path)
class_names = model.names
print("Model loaded successfully! Class names:", class_names)


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file!")
    exit()

frame_skip = 3  # Skip every 3rd frame for better speed
count = 0

while True:
    ret, img = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    count += 1
    if count % frame_skip != 0:
        continue  # Skip some frames to improve speed

    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape

    # ✅ Run YOLOv8 prediction
    results = model.predict(img)

    for r in results:
        boxes = r.boxes  # Bounding boxes

        # ✅ Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
            d = int(box.cls)  # Get class ID
            c = class_names[d] if d < len(class_names) else "Unknown"

            # ✅ Draw rectangle and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
            cv2.putText(img, c, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ✅ Show video output
    cv2.imshow("YOLOv8 Detection", img)

    # ✅ Slow down video (adjust wait time)
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# ✅ Release resources
cap.release()
cv2.destroyAllWindows()
