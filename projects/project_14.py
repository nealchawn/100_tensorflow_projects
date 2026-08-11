import tensorflow_datasets as tfds
import tensorflow as tf

"""
Project 14: TensorFlow Datasets (tfds) Loader
Description:
Use tensorflow_datasets (TFDS) to easily load and preprocess standard datasets like mnist, cifar10, or rock_paper_scissors with built-in train/test splits.

"""


# Load the CIFAR-10 dataset using TFDS
(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',                             # Dataset name
    split=['train', 'test'],               # Specify train/test splits
    shuffle_files=True,                    # Shuffle data files
    as_supervised=True,                    # Return (image, label) pairs
    with_info=True                         # Get metadata about the dataset
)
 
# Normalize and batch the data
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0            # Normalize image pixels to [0, 1]
    return image, label
 
batch_size = 32
ds_train = ds_train.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)  # Map, batch, prefetch
ds_test = ds_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
 
# Build a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),  # Conv layer
    tf.keras.layers.MaxPooling2D(),                                             # Pooling layer
    tf.keras.layers.Flatten(),                                                  # Flatten before Dense
    tf.keras.layers.Dense(64, activation='relu'),                               # Hidden layer
    tf.keras.layers.Dense(10, activation='softmax')                             # Output for 10 classes
])
 
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
# Train the model
model.fit(ds_train, epochs=5, validation_data=ds_test, verbose=1)