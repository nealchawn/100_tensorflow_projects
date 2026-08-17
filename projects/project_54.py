import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

"""
Project 54: Speech Emotion Recognition (MFCC + DNN)
Description:
Extract MFCC features from audio and train a deep neural network to classify emotions (e.g., happy, sad, angry) from speech signals.
"""

# Example audio file URL (RAVDESS dataset sample)
url = "https://github.com/pyannote/pyannote-audio/raw/develop/tutorials/assets/sample.wav"
path = tf.keras.utils.get_file("sample.wav", origin=url)
 
# Load audio file and extract MFCCs
y, sr = librosa.load(path, sr=16000)                      # Load with sampling rate 16kHz
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)       # Extract 13 MFCCs
mfccs = mfccs.T                                           # Shape: (time, features)
 
# Simulate multiple samples by repeating and adding noise
X = np.stack([mfccs + np.random.normal(scale=0.1, size=mfccs.shape) for _ in range(100)])
y_labels = np.random.randint(0, 3, 100)                   # 3 emotion classes (e.g., happy, sad, angry)
 
# Pad or truncate MFCC sequences to fixed length
max_len = 100
X_padded = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')
 
# Convert labels to one-hot
y_cat = tf.keras.utils.to_categorical(y_labels, num_classes=3)
 
# Build DNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_len, 13)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 emotion classes
])
 
# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_padded, y_cat, epochs=10, validation_split=0.2, verbose=0)
 
# Predict a single example
pred = model.predict(X_padded[:1])[0]
emotion = np.argmax(pred)
print("Predicted Emotion Class:", emotion)
