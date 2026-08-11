import tensorflow as tf
import matplotlib.pyplot as plt

"""
Project 17: Image Augmentation with tf.image
Description:
Use tf.image to apply real-time data augmentation techniques like flipping, rotating, and adjusting contrast to improve generalization in image classification.
"""

# Load and preprocess a sample image
image_path = tf.keras.utils.get_file(
    "puppy.jpg", 
    "https://storage.googleapis.com/download.tensorflow.org/example_images/puppy.jpg"
)
 
img_raw = tf.io.read_file(image_path)                          # Read image file as bytes
img = tf.image.decode_jpeg(img_raw, channels=3)                # Decode JPEG image
img = tf.image.resize(img, [224, 224])                         # Resize to 224x224
img = tf.cast(img, tf.float32) / 255.0                         # Normalize to [0, 1]
 
# Define a list of augmentations using tf.image
augmented_images = [
    img,                                                       # Original image
    tf.image.flip_left_right(img),                             # Horizontal flip
    tf.image.flip_up_down(img),                                # Vertical flip
    tf.image.rot90(img),                                       # Rotate 90 degrees
    tf.image.adjust_brightness(img, 0.3),                      # Brightness adjustment
    tf.image.adjust_contrast(img, 2.0),                        # Contrast enhancement
    tf.image.random_crop(tf.image.resize(img, [256, 256]), size=[224, 224, 3])  # Random crop
]
 
# Plot all augmented images
plt.figure(figsize=(12, 6))
for i, augmented in enumerate(augmented_images):
    plt.subplot(2, 4, i + 1)
    plt.imshow(tf.clip_by_value(augmented, 0.0, 1.0))          # Clip values to [0, 1] for display
    plt.axis('off')
    plt.title(f"Augmentation {i}")
plt.tight_layout()
plt.show()
