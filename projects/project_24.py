import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

"""
Project 24: Semantic Segmentation with DeepLabV3+
Description:
Use a pretrained DeepLabV3+ model from TensorFlow Hub to perform semantic segmentation on a single image, highlighting different object classes.
"""


# Load the DeepLabV3+ model from TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/deeplabv3/1")
 
# Load and preprocess the input image
image_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/segmentation_input.jpg"
image_path = tf.keras.utils.get_file("street.jpg", origin=image_url)
 
img_raw = tf.io.read_file(image_path)                           # Read image bytes
img = tf.image.decode_jpeg(img_raw)                             # Decode JPEG
original_size = tf.shape(img)[:2]                               # Save original size for resizing output
img = tf.image.convert_image_dtype(img, tf.uint8)               # Ensure image is uint8
img_resized = tf.image.resize(img, [513, 513])                  # Resize to model's expected input
img_tensor = tf.expand_dims(img_resized, 0)                     # Add batch dimension
 
# Run the model
result = model(img_tensor)
segmentation_map = tf.argmax(result['semantic_pred'], axis=3)[0]  # Get class prediction map
 
# Define a color map (Pascal VOC colormap — simplified)
def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    for i in range(256):
        r, g, b = 0, 0, 0
        c = i
        for j in range(8):
            r |= (c & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        colormap[i] = [r, g, b]
    return colormap
 
# Convert class map to color map
colormap = create_pascal_label_colormap()
segmentation_color = tf.gather(colormap, segmentation_map)
 
# Resize segmentation map back to original image size
segmentation_color = tf.image.resize(segmentation_color, original_size, method='nearest')
 
# Plot the original image and segmentation result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(tf.image.decode_jpeg(img_raw))
plt.title("Original Image")
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.imshow(tf.cast(segmentation_color, tf.uint8))
plt.title("Segmented Image")
plt.axis('off')
plt.tight_layout()
plt.show()
