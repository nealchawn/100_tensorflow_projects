import tensorflow as tf
import matplotlib.pyplot as plt

"""
Project 31: Text Classification with IMDB
Description:
Train a binary text classification model using TensorFlow 2 to classify IMDB movie reviews as positive or negative.
"""

# Load the IMDB dataset (pre-tokenized, 10k word vocab)
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
 
# Pad sequences to the same length
maxlen = 200
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)
 
# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen),  # Word embeddings
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),                              # Bidirectional LSTM
    tf.keras.layers.Dense(64, activation='relu'),                                         # Dense layer
    tf.keras.layers.Dense(1, activation='sigmoid')                                        # Output: 0 (neg) or 1 (pos)
])
 
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
# Train the model
history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
 
# Evaluate on test data
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.2f}")
 
# Plot training/validation accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("IMDB Sentiment Classification")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()