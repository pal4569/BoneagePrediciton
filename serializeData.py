import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Load the CSV file containing the metadata
train_df = pd.read_csv('data/boneage-training-dataset.csv')
print(train_df.head())

def load_image(image_id, base_path='data/boneage-training-dataset/boneage-training-dataset'):
    file_path = os.path.join(base_path, f'{image_id}.png')
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)  
    image = tf.image.resize(image, [128, 128])  # Resize image to a fixed size
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# Load all images into a list and preprocess them
images = []
ages = []
genders = []

# Use tqdm to add a progress bar to the loop
for idx, row in tqdm(train_df.iterrows(), total=train_df.shape[0], desc="Loading images"):
    image_id = row['id']
    boneage = row['boneage']
    male = row['male']
    
    image = load_image(image_id)
    
    images.append(image)
    ages.append(boneage)
    genders.append(male)

# Convert lists to NumPy arrays
images = np.array(images)
ages = np.array(ages)
genders = np.array(genders)

print(f'Images shape: {images.shape}')
print(f'Ages shape: {ages.shape}')
print(f'Genders shape: {genders.shape}')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val, genders_train, genders_val = train_test_split(images, ages, genders, test_size=0.2, random_state=42)

# Save NumPy arrays to disk
np.savez('boneage_train.npz', images=X_train, ages=y_train, genders=genders_train)
np.savez('boneage_val.npz', images=X_val, ages=y_val, genders=genders_val)
