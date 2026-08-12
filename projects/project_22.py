import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

"""
Project 22: Object Detection with TF Hub SSD
Description:
Use a pretrained SSD MobileNet model from TensorFlow Hub to detect objects in an image, and draw bounding boxes and labels.
"""


# Load the SSD MobileNet V2 model from TF Hub
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
 
# COCO label map (first 91 labels)
labels_path = tf.keras.utils.get_file(
    'mscoco_label_map.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/mscoco_label_map.txt'
)
labels = [line.strip() for line in open(labels_path).readlines()]
 
# Load and preprocess the input image
image_url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Toco_Toucan_RWD.jpg"
image_path = tf.keras.utils.get_file("toucan.jpg", origin=image_url)
 
img_raw = tf.io.read_file(image_path)                          # Read image file
img = tf.image.decode_jpeg(img_raw, channels=3)                # Decode JPEG
img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]  # Normalize and add batch dim
 
# Run object detection
result = detector(img)
result = {key: value.numpy() for key, value in result.items()} # Convert tensors to numpy arrays
 
# Plot the image and draw bounding boxes
image_np = img[0].numpy()
plt.figure(figsize=(10, 6))
plt.imshow(image_np)
ax = plt.gca()
 
# Draw bounding boxes and labels
for i in range(len(result["detection_scores"])):
    score = result["detection_scores"][i]
    if score < 0.5:
        continue
    box = result["detection_boxes"][i]
    class_id = int(result["detection_classes"][i])
    label = labels[class_id]
 
    # Unpack box coordinates and scale to image size
    ymin, xmin, ymax, xmax = box
    h, w, _ = image_np.shape
    (left, right, top, bottom) = (xmin * w, xmax * w, ymin * h, ymax * h)
 
    # Draw box
    ax.add_patch(plt.Rectangle((left, top), right - left, bottom - top,
                               edgecolor='red', facecolor='none', linewidth=2))
    ax.text(left, top - 5, f"{label}: {score:.2f}", color='red', fontsize=10, backgroundcolor='white')
 
plt.axis('off')
plt.title("Object Detection with SSD MobileNet")
plt.show()