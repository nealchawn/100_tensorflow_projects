import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Project 50: Sequence Prediction with Conv1D
Description:
Use a 1D convolutional neural network (Conv1D) to perform time series forecasting by learning patterns from past sequence data.
"""

# Generate synthetic data: noisy sine wave
def generate_series(size=500):
    x = np.linspace(0, 50, size)
    y = np.sin(x) + np.random.normal(scale=0.1, size=size)
    return y.astype(np.float32)
 
series = generate_series()
 
# Normalize the series
series = (series - np.mean(series)) / np.std(series)
 
# Create input-output sequence pairs
def create_dataset(series, window_size=30):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X)[..., np.newaxis], np.array(y)
 
window_size = 30
X, y = create_dataset(series, window_size)
 
# Split into training and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
 
# Build Conv1D model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window_size, 1)),
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1)
])
 
# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
 
# Predict on test set
preds = model.predict(X_test[:100]).flatten()
 
# Plot predictions
plt.plot(y_test[:100], label='True')
plt.plot(preds, label='Predicted')
plt.title("Time Series Forecasting with Conv1D")
plt.xlabel("Step")
plt.ylabel("Normalized Value")
plt.legend()
plt.show()