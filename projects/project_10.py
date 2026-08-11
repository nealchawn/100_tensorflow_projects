import tensorflow as tf
import matplotlib.pyplot as plt


"""
Project 10: Image Classification with Fashion MNIST
Description:
Train a neural network to classify clothing images images from the Fashion MNIST dataset using TensorFlow 2.
"""

 
# Load the Fashion MNIST dataset from Keras
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
 
# Normalize pixel values to range [0, 1]
X_train = X_train / 255.0                                       # Scale training images
X_test = X_test / 255.0                                         # Scale test images
 
# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),             # Flatten 28x28 images to 784
    tf.keras.layers.Dense(128, activation='relu'),             # Hidden layer with 128 ReLU units
    tf.keras.layers.Dense(10, activation='softmax')            # Output layer for 10 classes
])
 
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.1, verbose=1)  # Include validation split
 
# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.2f}")
 
# Optional: Display sample predictions
preds = model.predict(X_test[:5])                            # Predict on first 5 test images
for i, pred in enumerate(preds):
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"Predicted: {pred.argmax()}, True: {y_test[i]}")
    plt.axis('off')
    plt.show()