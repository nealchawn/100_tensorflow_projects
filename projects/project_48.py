import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Project 48: Multivariate Time Series Prediction
Description:
Build an LSTM model to forecast temperature using multiple weather features (e.g., pressure, humidity, wind speed) from the Jena Climate dataset.
"""

# Load Jena Climate dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/jena_climate_2009_2016.csv"
path = tf.keras.utils.get_file("jena_climate.csv", origin=url)
df = pd.read_csv(path)
 
# Select multiple features
features = df[["T (degC)", "p (mbar)", "rh (%)", "wv (m/s)"]].values.astype(np.float32)
 
# Normalize each feature
mean = features.mean(axis=0)
std = features.std(axis=0)
features = (features - mean) / std
 
# Create sequences (past 24 time steps → next temperature)
def create_multivariate_dataset(data, target_index=0, window=24):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window, target_index])  # Predict temperature
    return np.array(X), np.array(y)
 
X, y = create_multivariate_dataset(features, target_index=0, window=24)
 
# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
 
# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(24, X.shape[2])),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1)
])
 
# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5, validation_split=0.2)
 
# Predict and plot
preds = model.predict(X_test[:100]).flatten()
plt.plot(y_test[:100], label='True Temp')
plt.plot(preds, label='Predicted Temp')
plt.title("Multivariate Time Series Forecasting")
plt.xlabel("Hour")
plt.ylabel("Normalized Temperature")
plt.legend()
plt.show()