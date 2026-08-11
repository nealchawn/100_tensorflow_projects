import tensorflow as tf
import matplotlib.pyplot as plt

"""
Project 11: Image Preprocessing and Display
Description:
Load an image from a URL, preprocess it (resize and normalize), and display the processed image
Create a TensorFlow image pipeline that reads, decodes, resizes, and normalizes images to feed into a neural network model.
"""

 
# Example image path (you can replace this with your own image path)
image_path = tf.keras.utils.get_file(
    "grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
)
 
# Function to load and preprocess an image
def preprocess_image(path):
    img_raw = tf.io.read_file(path)                         # Read image file as binary
    img = tf.image.decode_jpeg(img_raw, channels=3)         # Decode JPEG to tensor with 3 color channels
    img = tf.image.resize(img, [224, 224])                  # Resize to 224x224
    img = tf.cast(img, tf.float32) / 255.0                  # Normalize pixel values to [0, 1]
    return img
 
# Preprocess the image
image = preprocess_image(image_path)
 
# Display the processed image
plt.imshow(image)
plt.title("Normalized Image")
plt.axis('off')
plt.show()