import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Project 53: Air Quality Prediction using LSTM
Description:
Use a multivariate LSTM model to predict future air quality index (AQI) based on historical pollutant concentrations (e.g., PM2.5, NO₂, CO).
"""

# Generate synthetic air quality data (PM2.5, NO2, CO, AQI)
def generate_air_data(size=500):
    t = np.linspace(0, 50, size)
    pm25 = 50 + 10 * np.sin(0.3 * t) + np.random.normal(0, 5, size)
    no2 = 30 + 5 * np.cos(0.2 * t) + np.random.normal(0, 3, size)
    co = 1 + 0.2 * np.sin(0.1 * t) + np.random.normal(0, 0.1, size)
    aqi = 0.5 * pm25 + 0.3 * no2 + 20 * co + np.random.normal(0, 2, size)
    return np.stack([pm25, no2, co, aqi], axis=1).astype(np.float32)
 
data = generate_air_data()
 
# Normalize features
mean = data.mean(axis=0)
std = data.std(axis=0)
data = (data - mean) / std
 
# Create sequences (past 24 steps → next AQI)
def create_dataset(data, window=24, target_index=3):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window, target_index])
    return np.array(X), np.array(y)
 
X, y = create_dataset(data, window=24, target_index=3)
 
# Split dataset
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
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
 
# Predict and plot
preds = model.predict(X_test[:100]).flatten()
plt.plot(y_test[:100], label='True AQI')
plt.plot(preds, label='Predicted AQI')
plt.title("Air Quality Prediction (AQI)")
plt.xlabel("Hour")
plt.ylabel("Normalized AQI")
plt.legend()
plt.show()
