import tensorflow as tf
import numpy as np
import librosa
import random

"""
Project 57: Music Genre Classification using CNN
Description:
Classify music clips into genres (e.g., classical, jazz, rock) by extracting MFCC features and feeding them into a CNN.
"""

# Simulate MFCC features for 3 genres
def generate_genre_mfccs(genres=3, samples_per_genre=50, max_len=130, n_mfcc=20):
    X, y = [], []
    for genre in range(genres):
        for _ in range(samples_per_genre):
            freq = random.uniform(100, 800) if genre == 0 else random.uniform(400, 1200)
            signal = np.sin(np.linspace(0, 2 * np.pi * freq, 22050))  # Simulated genre tones
            mfcc = librosa.feature.mfcc(y=signal.astype(np.float32), sr=22050, n_mfcc=n_mfcc)
            mfcc = mfcc.T[:max_len]
            if mfcc.shape[0] < max_len:
                pad = max_len - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode='constant')
            X.append(mfcc)
            y.append(genre)
    return np.array(X), tf.keras.utils.to_categorical(y, num_classes=genres)
 
# Generate dataset
X, y = generate_genre_mfccs()
X = X[..., np.newaxis]  # Add channel dimension for CNN
 
# Train/test split
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
    tf.keras.layers.Dense(y.shape[1], activation='softmax')
])
 
# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🎵 Music Genre Classification Accuracy: {acc:.2f}")