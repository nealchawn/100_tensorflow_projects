import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

"""
Project 20: Transfer Learning with MobileNet
Description:
Use MobileNetV2 from TensorFlow Hub as a fixed feature extractor and train a custom classifier on a small dataset (e.g. cats vs dogs).
"""


# Load cats_vs_dogs dataset from TensorFlow Datasets
(train_ds, val_ds), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)
 
# Preprocess: resize and normalize
IMG_SIZE = 160
def format_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))     # Resize to 160x160
    image = image / 255.0                                    # Normalize to [0, 1]
    return image, label
 
train_ds = train_ds.map(format_image).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(format_image).batch(32).prefetch(tf.data.AUTOTUNE)
 
# Load MobileNetV2 as feature extractor (frozen)
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(
    feature_extractor_url, input_shape=(IMG_SIZE, IMG_SIZE, 3), trainable=False
)
 
# Build the model
model = tf.keras.Sequential([
    feature_extractor_layer,                     # Frozen pretrained feature extractor
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])
 
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
# Train the model
model.fit(train_ds, validation_data=val_ds, epochs=3)
 
# Show prediction on one image
for image, label in val_ds.take(1):
    pred = model.predict(image[:1])
    plt.imshow(image[0])
    plt.title(f"Predicted: {'Dog' if pred[0][0] > 0.5 else 'Cat'}")
    plt.axis('off')
    plt.show()
    break