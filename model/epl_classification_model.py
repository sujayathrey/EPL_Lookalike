### Mobile Net ###

import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNet # type: ignore
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.models import Model # type: ignore
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('epl_player_data_224x224_60.csv')
data['label'] = data['label'].astype(str)  # Ensure labels are strings

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# Split the dataset into training and testing
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create data generators
train_generator = train_datagen.flow_from_dataframe(
    train_data,
    x_col='img_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    test_data,
    x_col='img_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Build the model using MobileNet
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
out = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=test_generator.samples // BATCH_SIZE
)

# Save the model
#model.save('epl_player_classification_model_mobilenet_224x224.h5')
model.save('epl_player_mobilenet_model.keras')

# Evaluate the model on the test set
results = model.evaluate(test_generator)
print(f"Test Loss: {results[0]} | Test Accuracy: {results[1]}")

