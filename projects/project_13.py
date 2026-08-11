import tensorflow as tf
import datetime

"""
Project 14: TensorBoard with MNIST CNN
Description:
Train a simple CNN on the MNIST dataset and visualize training metrics using TensorBoard.
"""

# Load and normalize MNIST dataset
(X_train, y_train), _ = tf.keras.datasets.mnist.load_data()        # Use only training data
X_train = X_train / 255.0                                          # Normalize pixel values
X_train = X_train[..., tf.newaxis]                                 # Add channel dimension (28,28,1)
 
# Create a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),  # Conv layer
    tf.keras.layers.Flatten(),                                                 # Flatten for Dense layer
    tf.keras.layers.Dense(64, activation='relu'),                              # Hidden layer
    tf.keras.layers.Dense(10, activation='softmax')                            # Output layer
])
 
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
# Set up TensorBoard log directory
log_dir = "logs/tensorboard_demo/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 
# Create TensorBoard callback
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
 
# Train the model with TensorBoard callback
model.fit(X_train, y_train, epochs=5, validation_split=0.2, callbacks=[tensorboard_cb], verbose=0)
 
# To launch TensorBoard, run this command in a separate terminal (not in Python):
# tensorboard --logdir=logs/tensorboard_demo/

