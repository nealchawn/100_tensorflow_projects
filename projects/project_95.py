import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

"""
Project 95: Image Super-Resolution using Pretrained Models
Description:
Enhance the resolution of images using a pretrained super-resolution model like ESRGAN or a simpler SRGAN variant.
"""

# Load the pretrained super-resolution model (using a simple pre-trained model for demonstration)
# You can use models like ESRGAN or any other available super-resolution model for better results
model = load_model("path_to_super_resolution_model.h5")  # Replace with actual model path
 
# Read an image (low resolution)
image = cv2.imread("low_resolution_image.jpg")  # Replace with your low-res image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)
 
# Preprocess image (resize or pad)
image_resized = image_pil.resize((image_pil.width * 2, image_pil.height * 2))  # Upscale by factor of 2 for demo
 
# Convert image to array
image_array = np.array(image_resized) / 255.0  # Normalize
image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
 
# Perform super-resolution
high_res_image = model.predict(image_array)
 
# Postprocess and display the result
high_res_image = np.squeeze(high_res_image) * 255.0  # Denormalize
high_res_image = np.clip(high_res_image, 0, 255).astype("uint8")
 
# Show the result
high_res_pil = Image.fromarray(high_res_image)
high_res_pil.show()
 
# Save the high-resolution image
high_res_pil.save("high_resolution_image.jpg")