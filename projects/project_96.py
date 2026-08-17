import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape
from tensorflow.keras.models import Model

"""
Project 96: Image-to-Image Translation with GANs
Description:
Use a Generative Adversarial Network (GAN) for image-to-image translation (e.g., converting sketches to photos, day to night transformation). We'll simulate the architecture and process here using a simple model like CycleGAN or Pix2Pix.
"""

# Define the Generator Model (for image-to-image translation)
def build_generator():
    model = tf.keras.Sequential([
        Input(shape=(256, 256, 3)),
        Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(256 * 256 * 3, activation='tanh'),
        Reshape((256, 256, 3))
    ])
    return model
 
# Define the Discriminator Model (to classify fake vs real images)
def build_discriminator():
    model = tf.keras.Sequential([
        Input(shape=(256, 256, 3)),
        Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model
 
# Build models
generator = build_generator()
discriminator = build_discriminator()
 
# Compile the discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
# GAN Model (combines generator and discriminator)
input_img = Input(shape=(256, 256, 3))
generated_img = generator(input_img)
discriminator.trainable = False  # Freeze discriminator during generator training
validity = discriminator(generated_img)
 
gan_model = Model(input_img, validity)
gan_model.compile(optimizer='adam', loss='binary_crossentropy')
 
# Generate a sample random image (for demonstration purposes)
sample_input = np.random.randn(1, 256, 256, 3)  # Random noise input (replace with actual image data)
generated_img = generator.predict(sample_input)
 
# Plot generated image
plt.imshow(generated_img[0])
plt.title("Generated Image (Image-to-Image Translation)")
plt.axis('off')
plt.show()
