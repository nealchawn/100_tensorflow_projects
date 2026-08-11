import tensorflow as tf
import os

"""
Project 12: Train/Validation/Test Split Utility
Description:

Implement a utility function using TensorFlow's tf.data API to split image data into training, validation, and test sets from a directory.

"""
 
# Load image dataset from directory (organized by class subfolders)
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=dataset_url, extract=True)
data_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered', 'train')  # Use train set
 
# Function to create split datasets
def create_split_datasets(data_dir, img_size=(160, 160), batch_size=32, val_split=0.2, test_split=0.1):
    # First, split into training and validation
    full_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=val_split + test_split,             # Reserve both val + test initially
        subset="training",
        seed=123
    )
    
    valtest_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=val_split + test_split,
        subset="validation",
        seed=123
    )
    
    val_batches = int(val_split / (val_split + test_split) * len(valtest_dataset))  # Split into val and test
    
    val_dataset = valtest_dataset.take(val_batches)            # First part = validation set
    test_dataset = valtest_dataset.skip(val_batches)           # Remaining part = test set
    
    return full_dataset, val_dataset, test_dataset
 
# Generate datasets
train_ds, val_ds, test_ds = create_split_datasets(data_dir)
 
# Print dataset sizes
print(f"Train batches: {len(train_ds)}")
print(f"Validation batches: {len(val_ds)}")
print(f"Test batches: {len(test_ds)}")