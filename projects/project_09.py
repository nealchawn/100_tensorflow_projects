import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Project 9: Binary Classification on Tabular Data
Description:

Train a binary classification model on the Pima Indians Diabetes dataset using TensorFlow 2 and evaluate its accuracy.
"""

 
# Load Pima Indians Diabetes dataset from UCI (via URL)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, names=cols)
 
# Split features and target
X = df.drop("Outcome", axis=1).values                    # All columns except 'Outcome' are features
y = df["Outcome"].values                                 # Binary target: 0 or 1
 
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# Build a binary classification model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=[X.shape[1]]),  # Hidden layer
    tf.keras.layers.Dense(8, activation='relu'),                             # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')                           # Output layer for binary prob
])
 
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
# Train the model
model.fit(X_train, y_train, epochs=100, validation_split=0.1, verbose=0)     # Train silently
 
# Evaluate on test set
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.2f}")