import tensorflow as tf
import numpy as np

"""
Project 25: OCR with CNN + CTC Loss (Handwritten Text Recognition)
Description:
Implement a simplified Optical Character Recognition (OCR) pipeline using a CNN and CTC (Connectionist Temporal Classification) loss for sequence prediction of handwritten words (e.g., from IAM or synthetic datasets).
"""


# Simulate data (normally you'd load from real OCR dataset like IAM or SynthText)
# Example: batch of 16 grayscale images of shape (100, 32) representing word images
batch_size = 16
img_width, img_height = 100, 32
num_classes = 26 + 1 + 1  # a-z + blank + padding/CTC special
X_data = np.random.rand(batch_size, img_height, img_width, 1).astype(np.float32)
y_data = np.random.randint(1, 27, size=(batch_size, 10))  # Random 10-letter sequences
 
# Input lengths (how many time steps per image) after downsampling (e.g., 1/4)
input_lengths = np.full((batch_size, 1), img_width // 4, dtype=np.int32)
label_lengths = np.full((batch_size, 1), 10, dtype=np.int32)
 
# Define a basic CNN + CTC model
input_img = tf.keras.Input(shape=(img_height, img_width, 1), name='input_image')
x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
 
# Reshape for RNN
new_shape = (img_width // 4, (img_height // 4) * 64)
x = tf.keras.layers.Reshape(target_shape=new_shape)(x)
 
# Bidirectional LSTM for sequence modeling
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # Predict char at each timestep
 
# Define CTC loss layer
labels = tf.keras.Input(shape=(None,), dtype='int32', name='labels')
input_len = tf.keras.Input(shape=(1,), dtype='int32', name='input_length')
label_len = tf.keras.Input(shape=(1,), dtype='int32', name='label_length')
 
def ctc_lambda_func(args):
    y_pred, labels, input_len, label_len = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_len, label_len)
 
loss_out = tf.keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
    [x, labels, input_len, label_len])
 
# Compile model with dummy loss (loss computed inside Lambda layer)
model = tf.keras.Model(inputs=[input_img, labels, input_len, label_len], outputs=loss_out)
model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred})
 
# Train with synthetic data
model.fit(
    x={'input_image': X_data, 'labels': y_data, 'input_length': input_lengths, 'label_length': label_lengths},
    y=np.zeros((batch_size, 1)),  # Dummy target for CTC
    epochs=1
)
