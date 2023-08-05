#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import required libraries
import cv2
import numpy as np

# Load the YOLO model
weights_path = "C:/Users/91728/Downloads/archive (1)/yolov3.weights"
cfg_path = "C:/Users/91728/Downloads/archive (1)/yolov3.cfg"

# Get the names of the output layers
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
ln = [layer_names[i.item() - 1] for i in net.getUnconnectedOutLayers()]

# Open the video file for reading
video_path = "C:/Users/91728/Downloads/archive (1)/123.mp4"
cap = cv2.VideoCapture(video_path)

# Define colors for bounding boxes (you can add more colors if needed)
colors = np.random.uniform(0, 255, size=(len(ln), 3))

# Dictionary to store the person IDs and their corresponding colors
person_colors = {}

# Dictionary to store the starting positions of each person
person_start_positions = {}

# Initialize variables to track accuracy
total_persons = 0
correctly_detected = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(ln)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Assuming person class_id is 0
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

# Perform non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    for i in indices:
        i = i.item()  # Access the index directly without reassignment
        x, y, w, h = boxes[i]
        person_id = i

        if person_id not in person_colors:
            person_colors[person_id] = colors[len(person_colors) % len(colors)]

        color = person_colors[person_id]

        if person_id not in person_start_positions:
            person_start_positions[person_id] = (x + w // 2, y + h // 2)  # Store initial position

        # Draw red line from starting position to current position
        start_x, start_y = person_start_positions[person_id]
        cv2.line(frame, (start_x, start_y), (x + w // 2, y + h // 2), (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Update current position as new starting position for the next frame
        person_start_positions[person_id] = (x + w // 2, y + h // 2)

        # Increment total_persons and correctly_detected
        total_persons += 1
        correctly_detected += 1

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate accuracy
accuracy = correctly_detected / total_persons if total_persons > 0 else 0.0
print("Total Persons:", total_persons)
print("Correctly Detected:", correctly_detected)
print("Accuracy:", accuracy)

cap.release()
cv2.destroyAllWindows()


# In[ ]:




