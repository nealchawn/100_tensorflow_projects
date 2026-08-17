import tensorflow as tf
import numpy as np
import librosa
import random

"""
Project 59: Speech Command Recognition with CNN
Description:
Classify short spoken commands (like "yes", "no", "stop") using a CNN model trained on MFCC features extracted from 1-second audio clips.

⚠️ For a full-scale implementation, use the Speech Commands dataset. This example simulates a simplified version.
"""

# Simulate 3 command words: "yes", "no", "stop"
def generate_command_data(classes=3, samples_per_class=50, max_len=100, n_mfcc=13):
    X, y = [], []
    for label in range(classes):
        for _ in range(samples_per_class):
            base_freq = 400 + label * 100  # Slightly different tone per command
            signal = np.sin(np.linspace(0, 2 * np.pi * base_freq, 16000))  # 1-second audio
            mfcc = librosa.feature.mfcc(y=signal.astype(np.float32), sr=16000, n_mfcc=n_mfcc)
            mfcc = mfcc.T[:max_len]
            if mfcc.shape[0] < max_len:
                pad = max_len - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode='constant')
            X.append(mfcc)
            y.append(label)
    return np.array(X), tf.keras.utils.to_categorical(y, num_classes=classes)
 
# Generate dataset
X, y = generate_command_data()
X = X[..., np.newaxis]  # Add channel dim for Conv2D
 
# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
 
# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y.shape[1], activation='softmax')  # Command class prediction
])
 
# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🎙️ Speech Command Recognition Accuracy: {acc:.2f}")