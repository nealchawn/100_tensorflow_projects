import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Define the model architecture
model = Sequential([
    # input layer flatten 28x28 images
    Flatten(input_shape=(28,28)),
    # Hidden layer with 128 neurons and ReLU activation
    Dense(128, activation='relu'),
    # Output layer with 10 neurons (for 10 classes) and softmax activation
    Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(train_images, train_labels, epochs=10,
    batch_size=32, validation_data=(val_images, val_labels))

# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Make predictions
predictions = model.predict(test_images)