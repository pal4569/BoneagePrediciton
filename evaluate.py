import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load validation data from .npz file
val_data = np.load('boneage_val.npz')
X_val = val_data['images']
y_val = val_data['ages']
genders_val = val_data['genders']

print(f'Validation images shape: {X_val.shape}')
print(f'Validation ages shape: {y_val.shape}')
print(f'Validation genders shape: {genders_val.shape}')

# Combine ages and genders into a single array for labels
labels_val = np.stack([y_val, genders_val], axis=1)

# Create a TensorFlow dataset
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, labels_val))

# Batch the dataset
val_dataset = val_dataset.batch(32)

# Prefetch to improve performance
val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Load the model
loaded_model = load_model('boneage_cnn_model.h5')

# Evaluate the model on the validation dataset
val_loss, val_mae = loaded_model.evaluate(val_dataset)
print(f'Validation Loss: {val_loss}, Validation MAE: {val_mae}')
