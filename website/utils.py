import json
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
import cv2
from mtcnn.mtcnn import MTCNN

# Constants
IMG_SIZE = (224, 224)

# Load class indices from JSON
with open('../model/class_indices.json', 'r') as f:
    class_indices = json.load(f)
# Reverse the dictionary to map indices to class labels
index_to_label = {v: k for k, v in class_indices.items()}

def detect_faces_mtcnn(image):
    detector = MTCNN()
    results = detector.detect_faces(image)
    if results:
        x, y, width, height = results[0]['box']
        face_img = image[y:y+height, x:x+width]
        return face_img
    return None

def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect and crop the face
    face_img = detect_faces_mtcnn(image)
    if face_img is None:
        raise ValueError(f"No face detected in image: {image_path}")

    # Resize the face image to the target size
    resized_face_img = cv2.resize(face_img, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)

    # Convert to array and preprocess for model input
    image_array = img_to_array(resized_face_img)
    processed_image = preprocess_input(image_array)

    return np.expand_dims(processed_image, axis=0)  # Add batch dimension

def predict_image(image_path, model):
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Get predictions
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]

    # Map the predicted index to a class label
    predicted_label = index_to_label[predicted_class_index]

    return {"label": predicted_label, "confidence": confidence}
