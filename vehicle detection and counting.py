import cv2
from sort import *
import math
import numpy as np
from ultralytics import YOLO
import cvzone

# Load video
cap = cv2.VideoCapture('video.mp4')

# Load YOLO model
model = YOLO('yolov8n.pt')

# Load class names
classnames  = []
with open('classes.txt','r') as f:
    classnames = f.read().splitlines()

# SORT Tracker
tracker = Sort(max_age=20)

# Line for vehicle counting
line = [320, 350, 620, 350]
counter = []

while True:
    ret, frame = cap.read()

    # If video ends → restart
    if not ret:
        cap = cv2.VideoCapture('cars2.mp4')
        continue

    detections = np.empty((0, 5))

    # YOLO inference
    result = model(frame, stream=True)

    for info in result:
        for box in info.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            classindex = int(box.cls[0])

            conf = int(conf * 100)
            objectdetect = classnames[classindex]

            # FIXED CONDITION → Parentheses added
            if (objectdetect in ['car', 'bus', 'truck']) and conf > 60:

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                new_det = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, new_det))

    # SORT tracker update
    track_result = tracker.update(detections)

    # Draw counting line
    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 7)

    # Process each tracked object
    for track in track_result:
        x1, y1, x2, y2, track_id = map(int, track)

        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Draw tracking visuals
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cvzone.putTextRect(frame, f'{track_id}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)

        # Counting vehicles
        if line[0] < cx < line[2] and (line[1] - 20) < cy < (line[1] + 20):
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 15)
            if track_id not in counter:
                counter.append(track_id)

    # Display count
    cvzone.putTextRect(frame, f'Total Vehicles = {len(counter)}', [290, 34], thickness=4, scale=2.3, border=2)

    # Show frame
    cv2.imshow('Vehicle Counter', frame)
    cv2.waitKey(1)
