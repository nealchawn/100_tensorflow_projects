import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras.applications import vgg19

"""
Project 90: Style Transfer with Pretrained VGG19
Description:
Perform neural style transfer to apply the artistic style of one image (e.g., painting) to the content of another image (e.g., a photo) using a pretrained VGG19 model.
"""

# Load content and style images
content_image_path = 'path_to_content_image.jpg'  # Replace with your image path
style_image_path = 'path_to_style_image.jpg'      # Replace with your image path
 
content_image = kp_image.load_img(content_image_path)
style_image = kp_image.load_img(style_image_path)
 
# Preprocess the images for VGG19
def preprocess_image(img):
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img
 
content_array = preprocess_image(content_image)
style_array = preprocess_image(style_image)
 
# Set up the VGG19 model
model = vgg19.VGG19(weights='imagenet', include_top=False)
 
# Get the layers for style and content extraction
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
 
def get_model(layers):
    outputs = [model.get_layer(name).output for name in layers]
    return tf.keras.models.Model([model.input], outputs)
 
content_model = get_model(content_layers)
style_model = get_model(style_layers)
 
# Define loss functions and optimization
def compute_content_loss(content, generated):
    return tf.reduce_mean(tf.square(content - generated))
 
def compute_style_loss(style, generated):
    gram_style = tf.linalg.einsum('bijc,bijd->bcd', style, style)
    gram_generated = tf.linalg.einsum('bijc,bijd->bcd', generated, generated)
    return tf.reduce_mean(tf.square(gram_style - gram_generated))
 
# Combine style and content losses for total loss calculation
def compute_total_loss(content_weight=1.0, style_weight=1.0):
    content_loss = compute_content_loss(content_array, generated_image)
    style_loss = compute_style_loss(style_array, generated_image)
    total_loss = content_weight * content_loss + style_weight * style_loss
    return total_loss
 
# Perform optimization and transfer the style to the content image
generated_image = tf.Variable(content_array)  # Initialize with content image
optimizer = tf.optimizers.Adam(learning_rate=0.01)
 
for i in range(1000):
    with tf.GradientTape() as tape:
        total_loss = compute_total_loss()
    grads = tape.gradient(total_loss, generated_image)
    optimizer.apply_gradients([(grads, generated_image)])
    if i % 100 == 0:
        print(f"Step {i}, Loss {total_loss.numpy()}")
 
# Convert the generated image to displayable format
generated_image = generated_image.numpy()
generated_image = np.squeeze(generated_image, axis=0)
generated_image = np.clip(generated_image, 0, 255).astype('uint8')
 
# Display the result
plt.imshow(generated_image)
plt.title("Generated Image with Style Transfer")
plt.axis('off')
plt.show()
