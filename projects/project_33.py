import tensorflow as tf
import matplotlib.pyplot as plt

"""
Project 33: Bidirectional LSTM for Sentiment Analysis
Description:
Build a bidirectional LSTM model using TensorFlow 2 to perform sentiment classification on IMDB movie reviews (positive or negative).
"""

# Load IMDB dataset with top 10,000 most frequent words
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
 
# Pad sequences to a uniform length
maxlen = 300
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)
 
# Define the Bi-LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen),   # Word embeddings
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),                                # Bi-LSTM layer
    tf.keras.layers.Dense(64, activation='relu'),                                            # Dense hidden layer
    tf.keras.layers.Dropout(0.5),                                                            # Dropout for regularization
    tf.keras.layers.Dense(1, activation='sigmoid')                                           # Output for binary classification
])
 
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
# Train the model
history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
 
# Evaluate on test set
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.2f}")
 
# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Bidirectional LSTM - IMDB Sentiment")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()