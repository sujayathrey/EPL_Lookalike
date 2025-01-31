import os
import cv2
import numpy as np
from PIL import Image
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, 
    Rotate, Blur, Resize, RandomCrop
)
from albumentations.core.composition import OneOf
from retinaface import RetinaFace

# Paths
input_path = "EPL_Player_Images"
output_path = "EPL_Players_Preprocessed_224x224_60"
os.makedirs(output_path, exist_ok=True)

# Constants
TARGET_SIZE = (224, 224)
TARGET_COUNT = 60  # Target number of images per player

# Augmentor
augmentor = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.5),
    Rotate(limit=20, p=0.5),
    Blur(blur_limit=3, p=0.3),
    OneOf([
        Resize(224, 224, p=1),
        RandomCrop(224, 224, p=1)
    ], p=1)
])

# RetinaFace-based face detection
def detect_faces_retinaface(image):
    detections = RetinaFace.detect_faces(image)
    if detections:
        for _, det in detections.items():
            x1, y1, x2, y2 = map(int, det["facial_area"])
            face_img = image[y1:y2, x1:x2]  # Crop face region
            return face_img
    return None

# Process each player folder
for player_folder in os.listdir(input_path):
    player_path = os.path.join(input_path, player_folder)
    output_player_path = os.path.join(output_path, player_folder)
    os.makedirs(output_player_path, exist_ok=True)

    # Collect valid images
    images = []
    for img_file in os.listdir(player_path):
        img_path = os.path.join(player_path, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_file}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            face_img = detect_faces_retinaface(img)
            if face_img is not None:
                # Resize face image to the target size first using INTER_LANCZOS4
                resized_face_img = cv2.resize(face_img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
                # Add the resized image to the list
                images.append(resized_face_img)
        
        except Exception as e:
            print(f"Skipping invalid image {img_file} in {player_folder}: {e}")

    # Save original images (enhanced)
    for i, img in enumerate(images):
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(output_player_path, f"original_{i}.jpg"))

    # Augment images to reach 60 (might switch to 50)
    aug_count = 0
    while len(images) + aug_count < TARGET_COUNT:
        for img in images:
            augmented = augmentor(image=img)["image"]
            augmented_pil = Image.fromarray(augmented)
            augmented_pil.save(os.path.join(output_player_path, f"augmented_{aug_count}.jpg"))
            aug_count += 1
            if len(images) + aug_count >= TARGET_COUNT:
                break

    print(f"Completed preprocessing for {player_folder}: {len(os.listdir(output_player_path))} images")

print("Preprocessing completed for all players.")