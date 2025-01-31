import os
import pandas as pd
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Load the dataset
data = pd.read_csv('epl_player_data_224x224_60.csv')
data['label'] = data['label'].astype(str)  # Ensure labels are strings

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create data generator
train_generator = train_datagen.flow_from_dataframe(
    data,
    x_col='img_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Save class indices
class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)

print("Class indices saved to class_indices.json.")