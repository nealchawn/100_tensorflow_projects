import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Project 47: Stock Price Prediction using RNN
Description:
Use a simple RNN model in TensorFlow 2 to predict future stock prices based on past closing values using a sliding window approach.
"""

# Load example stock price data (Apple from Yahoo Finance via pandas_datareader or CSV)
url = "https://raw.githubusercontent.com/selva86/datasets/master/aapl.csv"
df = pd.read_csv(url)
 
# Use the 'Close' column and normalize it
prices = df['Close'].values.astype(np.float32)
prices = (prices - prices.mean()) / prices.std()  # Standardize
 
# Create sequences (past 30 days → next day)
def create_dataset(series, window=30):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)
 
window_size = 30
X, y = create_dataset(prices, window=window_size)
X = X[..., np.newaxis]  # Add channel dimension
 
# Split data into training and testing sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
 
# Build simple RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, input_shape=(window_size, 1)),
    tf.keras.layers.Dense(1)
])
 
# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
 
# Predict on test set
preds = model.predict(X_test[:100]).flatten()
true = y_test[:100]
 
# Plot predictions vs true values
plt.plot(true, label='True Prices')
plt.plot(preds, label='Predicted Prices')
plt.title("Stock Price Prediction with RNN")
plt.xlabel("Days")
plt.ylabel("Normalized Price")
plt.legend()
plt.show()