import tensorflow as tf
import numpy as np
import librosa
import random

"""
Project 58: Audio Scene Classification (UrbanSound8K)
Description:
Classify urban sounds (e.g., sirens, dog barks, drilling) using MFCC features and a CNN trained on UrbanSound-like audio data.

⚠️ For a full project, use the UrbanSound8K dataset. Here we simulate the core pipeline.
"""

# Simulate audio features for 5 scene classes (e.g., siren, dog, drilling, engine, children)
def generate_urban_audio_data(classes=5, samples_per_class=40, max_len=100, n_mfcc=20):
    X, y = [], []
    for label in range(classes):
        for _ in range(samples_per_class):
            freq = random.uniform(200, 1000) + label * 50  # Vary frequency per class
            signal = np.sin(np.linspace(0, 2 * np.pi * freq, 22050))
            mfcc = librosa.feature.mfcc(y=signal.astype(np.float32), sr=22050, n_mfcc=n_mfcc)
            mfcc = mfcc.T[:max_len]
            if mfcc.shape[0] < max_len:
                pad = max_len - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode='constant')
            X.append(mfcc)
            y.append(label)
    return np.array(X), tf.keras.utils.to_categorical(y, num_classes=classes)
 
# Create dataset
X, y = generate_urban_audio_data()
X = X[..., np.newaxis]  # Add channel dimension
 
# Split into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
 
# Build CNN model for scene classification
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y.shape[1], activation='softmax')
])
 
# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🏙️ Urban Sound Scene Classification Accuracy: {acc:.2f}")