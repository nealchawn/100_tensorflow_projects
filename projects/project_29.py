import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Project 29: Image Denoising Autoencoder
Description:
Build a convolutional autoencoder using TensorFlow 2 to remove noise from images. We'll use MNIST digits with added Gaussian noise as the dataset.
"""

# Load MNIST and normalize
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train = np.expand_dims(X_train, -1)  # Add channel dimension
X_test = np.expand_dims(X_test, -1)
 
# Add Gaussian noise to images
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)
 
# Build convolutional autoencoder
input_img = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D(2)(x)
x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D(2)(x)
decoded = tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
 
autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
 
# Train the autoencoder
autoencoder.fit(X_train_noisy, X_train, epochs=5, batch_size=128, shuffle=True, validation_split=0.1)
 
# Predict denoised images
decoded_imgs = autoencoder.predict(X_test_noisy[:10])
 
# Display original noisy vs. denoised
plt.figure(figsize=(20, 4))
for i in range(10):
    # Noisy
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(X_test_noisy[i].squeeze(), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")
 
    # Denoised
    ax = plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_imgs[i].squeeze(), cmap="gray")
    plt.title("Denoised")
    plt.axis("off")
plt.show()