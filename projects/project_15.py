import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


"""
Project 15: Transfer Learning with Hub Pretrained Model Demo MobileNetV2
Description:
Use a pretrained MobileNetV2 model from TensorFlow Hub to classify images from ImageNet, a pretrained image classifier from TensorFlow Hub to perform transfer learning or inference on an input image.
"""


# Load a pretrained model from TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = tf.keras.Sequential([
    hub.KerasLayer(model_url, input_shape=(224, 224, 3))          # Load MobileNetV2 pretrained on ImageNet
])
 
# Load ImageNet labels for decoding predictions
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
)
imagenet_labels = np.array(open(labels_path).read().splitlines())  # Read labels into array
 
# Load and preprocess a sample image
image_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/puppy.jpg"
image_path = tf.keras.utils.get_file("puppy.jpg", origin=image_url)
 
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)                         # Read image file
    img = tf.image.decode_jpeg(img, channels=3)         # Decode JPEG
    img = tf.image.resize(img, [224, 224])              # Resize to 224x224
    img = img / 255.0                                   # Normalize pixel values
    return tf.expand_dims(img, axis=0)                  # Add batch dimension
 
image = load_and_preprocess_image(image_path)
 
# Predict class using the model
predictions = model(image)                              # Get prediction logits
predicted_class = tf.argmax(predictions[0]).numpy()     # Get index of highest probability class
predicted_label = imagenet_labels[predicted_class]      # Get class label from index
 
# Show result
plt.imshow(tf.squeeze(image))                           # Display the image
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()
