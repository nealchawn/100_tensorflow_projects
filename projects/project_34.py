import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import itertools
from collections import defaultdict

"""
Project 34: Word Embeddings with Word2Vec (Custom Training)
Description:
Train custom Word2Vec word embeddings using TensorFlow and a skip-gram model on a small text corpus.
"""

# Sample corpus
sentences = [
    "machine learning is fun",
    "deep learning is part of machine learning",
    "natural language processing is a field of ai",
    "word embeddings are learned representations",
    "tensorflow makes it easy to build models"
]
 
# Tokenize corpus
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
word2idx = tokenizer.word_index
idx2word = {v: k for k, v in word2idx.items()}
vocab_size = len(word2idx) + 1
 
# Generate skip-gram pairs
window_size = 2
sequences = tokenizer.texts_to_sequences(sentences)
pairs = []
for seq in sequences:
    for i, target_word in enumerate(seq):
        context_window = seq[max(i - window_size, 0): i] + seq[i + 1: i + window_size + 1]
        for context_word in context_window:
            pairs.append((target_word, context_word))
 
# Convert to numpy arrays
targets, contexts = zip(*pairs)
targets = np.array(targets)
contexts = np.array(contexts)
 
# One-hot encode targets
context_labels = tf.keras.utils.to_categorical(contexts, num_classes=vocab_size)
 
# Define skip-gram model
embedding_dim = 64
input_word = tf.keras.Input(shape=(1,))
embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_word)
x = tf.keras.layers.Reshape((embedding_dim,))(embedding)
output = tf.keras.layers.Dense(vocab_size, activation='softmax')(x)
 
model = tf.keras.Model(inputs=input_word, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy')
 
# Train the model
model.fit(targets, context_labels, epochs=100, verbose=0)
 
# Extract and display learned embeddings
embedding_weights = model.get_layer('embedding').get_weights()[0]
for word, idx in word2idx.items():
    vec = embedding_weights[idx][:5]  # Show first 5 dims
    print(f"{word}: {vec.round(3)}")