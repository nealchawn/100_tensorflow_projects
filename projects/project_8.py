import tensorflow as tf
import numpy as np
import os

"""
Project 8: Early Stopping & Model Checkpointing
Description:

Show how to use early stopping to prevent overfitting and save the best model using model checkpointing during training.
"""

# Create synthetic regression data
X = np.linspace(0, 10, 300).reshape(-1, 1).astype(np.float32)         # Inputs from 0 to 10
y = 2 * X + 3 + np.random.randn(*X.shape) * 0.5                       # Target with noise
 
# Define a simple feedforward model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[1]),   # Hidden layer
    tf.keras.layers.Dense(1)                                         # Output layer
])
 
# Compile with MSE loss and Adam optimizer
model.compile(optimizer='adam', loss='mse', metrics=['mae'])         # Mean Absolute Error as metric
 
# Setup callbacks: EarlyStopping and ModelCheckpoint
early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True) # Stop if no improvement
checkpoint_path = "best_model.h5"                                     # Path to save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True)
 
# Train with callbacks
history = model.fit(X, y, epochs=100, validation_split=0.2,
                    callbacks=[early_stop, checkpoint], verbose=0)
 
# Load best model weights (optional — already restored if early_stop.restore_best_weights=True)
model.load_weights(checkpoint_path)
 
# Evaluate final model
loss, mae = model.evaluate(X, y, verbose=0)
print(f"Final MAE: {mae:.3f}")