import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

"""
Project 30: Depth Estimation from Monocular Images
Description:
Use a pretrained monocular depth estimation model from TensorFlow Hub to predict a depth map from a single RGB image.
"""

# Load the MiDaS depth estimation model from TensorFlow Hub
model_url = "https://tfhub.dev/intel/midas/v2_1_small/1"
depth_model = hub.load(model_url)
 
# Preprocess image: resize, normalize, batch
def load_and_preprocess(image_url):
    image_path = tf.keras.utils.get_file("input.jpg", origin=image_url)
    img = Image.open(image_path).convert("RGB").resize((256, 256))       # Resize image
    img = np.array(img).astype(np.float32) / 255.0                       # Normalize
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, (256, 256))
    return tf.expand_dims(img, 0)                                        # Add batch dimension
 
# Load sample image
image_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"
input_image = load_and_preprocess(image_url)
 
# Run inference to get depth map
depth_map = depth_model(input_image)['default'][0]                       # Remove batch dimension
depth_map = tf.image.resize(depth_map[..., tf.newaxis], (256, 256))     # Resize for visualization
 
# Normalize depth for display
depth_min = tf.reduce_min(depth_map)
depth_max = tf.reduce_max(depth_map)
normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
 
# Show original image and predicted depth
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(tf.squeeze(input_image))
plt.title("Original Image")
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.imshow(tf.squeeze(normalized_depth), cmap='inferno')
plt.title("Estimated Depth Map")
plt.axis('off')
plt.show()