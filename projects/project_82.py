import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np

"""
Project 82: Multilabel Text Classification with Sigmoid Output
Description:
Build a multilabel classifier using TensorFlow and Keras, where each text can belong to multiple categories (e.g., "tech" and "finance").
"""

# Sample dataset (texts with multiple tags)
texts = [
    "The stock market saw major gains today.",               # finance
    "New iPhone features cutting-edge technology.",          # tech
    "Google and Microsoft both released earnings reports.",  # tech, finance
    "The Lakers won the basketball championship.",           # sports
    "Tesla's new AI chips are revolutionizing cars."         # tech, auto
]
 
labels = [
    ["finance"],
    ["tech"],
    ["tech", "finance"],
    ["sports"],
    ["tech", "auto"]
]
 
# Binarize multilabels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)
 
# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts).toarray()
 
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multilabel classification model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(mlb.classes_), activation='sigmoid')  # One sigmoid per label
])
 
# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=0)
 
# Predict on new sample
new_text = ["Apple's quarterly earnings beat expectations."]
new_X = vectorizer.transform(new_text).toarray()
pred = model.predict(new_X)[0]
thresholded = [mlb.classes_[i] for i, p in enumerate(pred) if p > 0.5]
 
print("📝 New Text:\n", new_text[0])
print("\n📌 Predicted Tags:", thresholded)