from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
from PIL import Image
import utils  # Your helper functions for preprocessing and prediction

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
MODEL_PATH = '../model/epl_player_mobilenet_model.keras'
model = load_model(MODEL_PATH)

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict using the model
        prediction = utils.predict_image(file_path, model)

        return render_template('result.html', prediction=prediction, image_path=file_path)
    else:
        return "File type not allowed. Please upload a PNG, JPG, or JPEG file."

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
