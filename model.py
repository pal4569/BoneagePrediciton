from tensorflow.keras import layers, models
import numpy as np
import tensorflow as tf
import json

EPOCHS = 10

# Load NumPy arrays from disk
data = np.load('boneage_train.npz')
images = data['images']
ages = data['ages']
genders = data['genders']

# Combine ages and genders into a single array for multi-input model
labels = np.stack([ages, genders], axis=1)

# Convert the NumPy arrays to TensorFlow datasets
dataset = tf.data.Dataset.from_tensor_slices((images, labels))

# Shuffle and batch the dataset
batch_size = 32
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prefetch to improve performance
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# This is a CNN model.
# I used an increasing number of filters to handle the increasing
# complexity of the features.
def create_model():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),  # Images are 128x128 px
        data_augmentation,  # Add data augmentation here
        layers.Conv2D(32, (3, 3), activation='relu'), # Basic edges and textures
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'), # More complex shapes
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'), # Even more complex shapes and patterns
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),  # Flatten image into "1D" array of px
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Predict bone age
    ])
    
    # Compiler initialized for regression task
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = create_model()
model.summary()

# Trains the model 
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)
model.save('boneage_cnn_model.h5')
with open('boneage_training_history.json', 'w') as f:
    json.dump(history.history, f)
