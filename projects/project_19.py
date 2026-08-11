import tensorflow as tf
import matplotlib.pyplot as plt
import os

"""
Project 19: Custom CNN for Face Mask Detection
Description:
Train a custom CNN on a face mask dataset to classify whether a person is wearing a mask or not. This uses TensorFlow’s image_dataset_from_directory API for loading data.

"""


# Dataset link (you'll need to manually download and unzip this in practice)
# Example dataset structure:
# dataset/
# ├── with_mask/
# └── without_mask/
dataset_dir = "/path/to/face_mask_dataset"  # Replace with actual path
 
# Load dataset using image_dataset_from_directory
dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    image_size=(128, 128),                         # Resize all images to 128x128
    batch_size=32,                                 # Batch size
    label_mode='int',                              # Integer labels (0 or 1)
    validation_split=0.2,                          # 80-20 split
    subset="both",                                 # Load both train and val splits
    seed=42
)
 
train_ds, val_ds = dataset                         # Unpack training and validation sets
 
# Prefetch for performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
 
# Define the custom CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),   # Normalize pixel values
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')                 # Binary classification
])
 
# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
# Train the model
model.fit(train_ds, validation_data=val_ds, epochs=10)
 
# Example prediction (optional)
for images, labels in val_ds.take(1):                              # Take one batch
    preds = model.predict(images)                                  # Get predictions
    plt.imshow(images[0].numpy().astype("uint8"))
    plt.title(f"Predicted: {'Mask' if preds[0] < 0.5 else 'No Mask'}")
    plt.axis('off')
    plt.show()
    break