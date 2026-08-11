import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

"""
Project 3: Multilayer Perceptron from Scratch
Description:
Create a basic feedforward neural network (MLP) using TensorFlow 2 to classify data from the Iris dataset (3 classes).
"""


# Load Iris dataset
iris = load_iris()
X = iris.data                                           # Features: sepal & petal length/width
y = iris.target.reshape(-1, 1)                          # Labels: 0, 1, 2
 
# One-hot encode labels for multi-class classification
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)                    # Convert to one-hot format
 
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)                      # Normalize features
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
 
# Define the MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),   # First hidden layer with 10 ReLU units
    tf.keras.layers.Dense(8, activation='relu'),                      # Second hidden layer with 8 ReLU units
    tf.keras.layers.Dense(3, activation='softmax')                    # Output layer with softmax for 3 classes
])
 
# Compile the model with categorical crossentropy and Adam optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# Train the model
model.fit(X_train, y_train, epochs=100, verbose=0)        # Train silently for 100 epochs
 
# Evaluate the model on the test set
loss, acc = model.evaluate(X_test, y_test, verbose=0)     # Evaluate model performance
print(f"Test Accuracy: {acc:.2f}")
