from flask import Flask, request, jsonify, render_template, Response
import torch
from PIL import Image
import numpy as np
import cv2
from preprocessing import preprocess_image  

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/harsh/Desktop/Projects/Traffic_Sign_detection/yolov5/runs/train/exp/weights/best.pt', force_reload=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Load the image from the file
        image = Image.open(file).convert('RGB')

        # Perform prediction
        results = model(image)  
        img_with_boxes = results.render()[0]  # Get image with bounding boxes

        # Convert the image to PNG format
        _, img_encoded = cv2.imencode('.png', img_with_boxes)
        img_bytes = img_encoded.tobytes()

        # Return the image as a response
        return Response(img_bytes, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

