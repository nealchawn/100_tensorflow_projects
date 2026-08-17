import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Project 36: GRU-based Language Model
Description:
Train a character-level GRU language model using TensorFlow 2 to generate text one character at a time, based on a short input sequence.
"""

# Load and prepare text (tiny Shakespeare excerpt)
text = "To be, or not to be, that is the question."
chars = sorted(set(text))                            # Unique characters
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = np.array(chars)
vocab_size = len(chars)
 
# Convert entire text to integer sequence
text_as_int = np.array([char2idx[c] for c in text])
 
# Create input-output pairs for training (seq → next char)
seq_length = 10
examples_per_epoch = len(text_as_int) - seq_length
inputs = []
targets = []
for i in range(examples_per_epoch):
    inputs.append(text_as_int[i:i+seq_length])
    targets.append(text_as_int[i+1:i+seq_length+1])
X = np.array(inputs)
y = np.array(targets)
 
# Build GRU language model
embedding_dim = 32
gru_units = 64
 
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=seq_length),  # Embedding layer
    tf.keras.layers.GRU(gru_units, return_sequences=True),                          # GRU layer
    tf.keras.layers.Dense(vocab_size, activation='softmax')                         # Output: char probabilities
])
 
# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=200, verbose=0)
 
# Generate text from a seed
def generate_text(seed, length=100):
    input_eval = [char2idx[c] for c in seed.lower()]
    input_eval = tf.expand_dims(input_eval, 0)
    result = list(seed)
 
    for _ in range(length):
        predictions = model(input_eval)
        predicted_id = tf.random.categorical(predictions[:, -1, :], num_samples=1).numpy()[0][0]
        result.append(idx2char[predicted_id])
        input_eval = tf.concat([input_eval[:, 1:], [[predicted_id]]], axis=1)
 
    return ''.join(result)
 
# Generate from seed
print(generate_text("To be, or "))