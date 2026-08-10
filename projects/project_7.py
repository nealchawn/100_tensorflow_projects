import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Project 7: Learning Rate Scheduling

Description:
Demonstrate how to use a learning rate scheduler in TensorFlow 2 
to adaptively reduce the learning rate during training for a regression task.
"""

# Generate synthetic regression data
X = np.linspace(0, 10, 200).reshape(-1, 1).astype(np.float32)      # Inputs: 0 to 10
y = 7 * X + 5 + np.random.randn(*X.shape) * 2                      # Targets: linear + noise
 
# Define a learning rate scheduler function
def scheduler(epoch, lr):
    if epoch % 20 == 0 and epoch:
        return lr * 0.5                                            # Halve learning rate every 20 epochs
    return lr
 
# Create a simple regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])               # Single neuron linear regressor
])
 
# Compile with Adam optimizer and MSE loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
 
# Use the scheduler as a callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
 
# Train the model with scheduler
history = model.fit(X, y, epochs=100, callbacks=[lr_callback], verbose=0)
 
# Predict and visualize results
preds = model.predict(X)
 
plt.scatter(X, y, label='True Data')                               # Original noisy data
plt.plot(X, preds, color='red', label='Predicted Line')            # Regression line
plt.title("Learning Rate Scheduling Demo")
plt.legend()
plt.show()
 
# Plot learning rate change over epochs
lrs = [scheduler(e, 0.01) for e in range(100)]
plt.plot(lrs)
plt.title("Learning Rate Schedule (step-wise decay)")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.show()