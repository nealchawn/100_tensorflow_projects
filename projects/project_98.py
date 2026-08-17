import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras.applications.vgg19 import preprocess_input
import matplotlib.pyplot as plt

"""
Project 98: Real-Time Video Style Transfer
Description:
Apply neural style transfer to video frames in real-time using a pre-trained model, transforming the video’s content into the style of an artwork (e.g., Van Gogh’s Starry Night).
"""

# Load the content (video) and style (painting) images
style_image_path = 'starry_night.jpg'  # Replace with style image path
content_video_path = 'input_video.mp4'  # Replace with video path
 
# Load the style image
style_image = kp_image.load_img(style_image_path)
style_image = kp_image.img_to_array(style_image)
style_image = np.expand_dims(style_image, axis=0)
style_image = preprocess_input(style_image)
 
# Build a VGG19 model for style and content extraction
vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
 
# Layers to use for style and content extraction
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layers = ['block5_conv2']
 
# Model to extract style and content
def get_model():
    outputs = [vgg.get_layer(layer).output for layer in style_layers + content_layers]
    model = tf.keras.Model([vgg.input], outputs)
    return model
 
# Function for style transfer
def style_transfer(frame, model, style_image):
    content_image = np.expand_dims(frame, axis=0)
    content_image = preprocess_input(content_image)
    
    # Calculate content and style loss here (omitting full implementation for brevity)
    return frame  # Placeholder for the final output of style transfer
 
# Open video
cap = cv2.VideoCapture(content_video_path)
 
# Define the output video writer
output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
 
# Read and process video frames
model = get_model()
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply style transfer to the frame
    stylized_frame = style_transfer(frame, model, style_image)
 
    # Write the processed frame to the output video
    out.write(stylized_frame)
 
    # Display the stylized frame (optional)
    cv2.imshow('Stylized Video', stylized_frame)
 
    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()