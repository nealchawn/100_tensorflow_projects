import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

"""
Project 23: Real-Time Object Detection with Webcam
Description:
Use TensorFlow Hub’s SSD MobileNet model to perform object detection on live webcam feed using OpenCV and draw bounding boxes with class labels.
⚠️ Requires OpenCV (pip install opencv-python) and a working webcam.
"""

# Load the pretrained SSD MobileNet model from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
 
# Load COCO labels
labels_path = tf.keras.utils.get_file(
    'mscoco_label_map.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/mscoco_label_map.txt'
)
labels = [line.strip() for line in open(labels_path).readlines()]
 
# Start webcam feed
cap = cv2.VideoCapture(0)  # Use camera index 0
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    # Preprocess the frame: resize, normalize, expand dims
    input_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # Convert BGR to RGB
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
    input_tensor = tf.image.resize(input_tensor, (320, 320)) / 255.0
    input_tensor = input_tensor[tf.newaxis, ...]
 
    # Run detection
    detections = detector(input_tensor)
    detections = {k: v.numpy() for k, v in detections.items()}
 
    # Draw detections with score > 0.5
    h, w, _ = frame.shape
    for i in range(len(detections['detection_scores'])):
        score = detections['detection_scores'][i]
        if score < 0.5:
            continue
 
        box = detections['detection_boxes'][i]
        class_id = int(detections['detection_classes'][i])
        label = labels[class_id]
 
        ymin, xmin, ymax, xmax = box
        left, top = int(xmin * w), int(ymin * h)
        right, bottom = int(xmax * w), int(ymax * h)
 
        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
 
    # Show frame
    cv2.imshow('Real-Time Object Detection', frame)
 
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()