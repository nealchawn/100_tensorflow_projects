import tensorflow as tf
import numpy as np
import librosa
import random

"""
Project 56: Speaker Identification using MFCC + LSTM
Description:
Train an LSTM model to identify speakers based on MFCC features extracted from their voice recordings.
"""

# Simulate loading MFCC features from 3 speakers
def generate_speaker_mfccs(num_speakers=3, samples_per_speaker=50, max_len=100, n_mfcc=13):
    data, labels = [], []
    for speaker_id in range(num_speakers):
        for _ in range(samples_per_speaker):
            signal = np.sin(np.linspace(0, 2 * np.pi * (random.uniform(100, 300)), 16000))  # Simulated tone
            mfcc = librosa.feature.mfcc(y=signal.astype(np.float32), sr=16000, n_mfcc=n_mfcc)
            mfcc = mfcc.T[:max_len]  # Trim/pad to max_len
            if mfcc.shape[0] < max_len:
                pad_width = max_len - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            data.append(mfcc)
            labels.append(speaker_id)
    return np.array(data), tf.keras.utils.to_categorical(labels, num_classes=num_speakers)
 
# Generate dataset
X, y = generate_speaker_mfccs()
 
# Split into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
 
# Build LSTM-based speaker ID model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),     # (max_len, n_mfcc)
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y.shape[1], activation='softmax')    # Output: num_speakers
])
 
# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
 
# Evaluate on test set
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🎤 Speaker ID Accuracy: {acc:.2f}")