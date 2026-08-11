import tensorflow as tf
import matplotlib.pyplot as plt

"""
Project 16: CNN for Handwritten Digit Recognition (MNIST)
Description:
Build and train a Convolutional Neural Network (CNN) using TensorFlow 2 to classify handwritten digits from the MNIST dataset.
"""


# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
 
# Normalize and reshape input images
X_train = X_train / 255.0                                # Normalize pixel values
X_test = X_test / 255.0
X_train = X_train[..., tf.newaxis]                       # Add channel dimension: (28, 28, 1)
X_test = X_test[..., tf.newaxis]
 
# Define CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),  # 32 filters, 3x3 kernel
    tf.keras.layers.MaxPooling2D(2, 2),                                             # Reduce spatial size
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),                           # Deeper feature extraction
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),                                                     # Flatten before Dense layer
    tf.keras.layers.Dense(64, activation='relu'),                                  # Hidden dense layer
    tf.keras.layers.Dense(10, activation='softmax')                                # Output layer (10 classes)
])
 
# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
# Train the CNN
model.fit(X_train, y_train, epochs=5, validation_split=0.1, verbose=1)
 
# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.2f}")
 
# Visualize predictions
preds = model.predict(X_test[:5])
for i, pred in enumerate(preds):
    plt.imshow(X_test[i].numpy().squeeze(), cmap='gray')
    plt.title(f"Predicted: {tf.argmax(pred).numpy()}, True: {y_test[i]}")
    plt.axis('off')
    plt.show()