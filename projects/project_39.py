import tensorflow as tf
import numpy as np

"""
Project 39: Transformer Encoder for Text Embeddings
Description:
Build a simplified Transformer Encoder using TensorFlow 2 and apply it to generate contextualized text embeddings for input tokens.
"""

# Sample tokenized sentence (already converted to integers)
sample_input = tf.constant([[3, 5, 7, 9, 0, 0]])  # Padded sequence of length 6
vocab_size = 20
maxlen = 6
embed_dim = 64
num_heads = 4
ff_dim = 128
 
# Positional Encoding Layer
class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        position = tf.range(seq_len, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]
        i = tf.range(embed_dim, dtype=tf.float32)[tf.newaxis, tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(embed_dim, tf.float32))
        angle_rads = position * angle_rates
        angle_rads[:, :, 0::2] = tf.math.sin(angle_rads[:, :, 0::2])
        angle_rads[:, :, 1::2] = tf.math.cos(angle_rads[:, :, 1::2])
        return inputs + tf.cast(angle_rads, tf.float32)
 
# Transformer Encoder Block
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
 
    def call(self, x):
        attn_output = self.att(x, x)                         # Self-attention
        out1 = self.norm1(x + attn_output)                   # Add & norm
        ffn_output = self.ffn(out1)                          # Feed-forward network
        return self.norm2(out1 + ffn_output)                 # Add & norm
 
# Define the full encoder model
inputs = tf.keras.Input(shape=(maxlen,))
x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)   # Word embeddings
x = PositionalEncoding()(x)                                                         # Add positional encoding
x = TransformerEncoder(embed_dim, num_heads, ff_dim)(x)                             # Transformer block
 
model = tf.keras.Model(inputs, x)
 
# Run the model on the sample input
output_embeddings = model(sample_input)
print("Output Embedding Shape:", output_embeddings.shape)
print("Token Embedding for first token:", output_embeddings[0, 0, :5].numpy().round(3))