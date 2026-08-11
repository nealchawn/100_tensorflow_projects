import tensorflow as tf
import matplotlib.pyplot as plt

"""
Project 18: CIFAR-10 Image Classification
Description:
Train a CNN model on the CIFAR-10 dataset to classify images into 10 categories like airplanes, cats, trucks, etc.
"""


# Load and normalize CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train / 255.0                                     # Normalize training images
X_test = X_test / 255.0                                       # Normalize test images
y_train = y_train.squeeze()                                   # Remove extra dimension
y_test = y_test.squeeze()
 
# Define class labels for display
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
 
# Build a CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),  # 32 filters
    tf.keras.layers.MaxPooling2D((2,2)),                                            # Max pooling
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),                           # 64 filters
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),                                                     # Flatten output
    tf.keras.layers.Dense(64, activation='relu'),                                  # Hidden dense layer
    tf.keras.layers.Dense(10, activation='softmax')                                # Output for 10 classes
])
 
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.1, verbose=1)
 
# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.2f}")
 
# Predict and display a few test images
preds = model.predict(X_test[:5])
for i, pred in enumerate(preds):
    plt.imshow(X_test[i])
    plt.title(f"Predicted: {class_names[tf.argmax(pred)]}\nTrue: {class_names[y_test[i]]}")
    plt.axis('off')
    plt.show()