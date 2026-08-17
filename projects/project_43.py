import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

"""
Project 43: Fake News Detection (TF-IDF + DNN)
Description:
Use a TF-IDF vectorizer and a deep neural network (DNN) to classify news headlines as real or fake using TensorFlow 2.
"""

# Sample dataset (headlines + labels)
texts = [
    "Donald Trump meets with the president of Mexico",
    "NASA announces the discovery of new exoplanets",
    "Breaking: Aliens found living in New York subway",
    "Obama caught hiding alien technology",
    "Scientists develop cure for cancer",
    "Elvis Presley spotted in Times Square"
]
labels = [0, 0, 1, 1, 0, 1]  # 0 = real, 1 = fake
 
# Split data
X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
 
# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_texts).toarray()
X_test = vectorizer.transform(X_test_texts).toarray()
 
# Build DNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output for binary classification
])
 
# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.2f}")
 
# Predict on a new headline
headline = ["New study reveals link between exercise and happiness"]
headline_vec = vectorizer.transform(headline).toarray()
pred = model.predict(headline_vec)[0][0]
print("Prediction:", "Fake" if pred > 0.5 else "Real")