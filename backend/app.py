from flask import Flask, request
from flask_cors import CORS
import os
from PIL import Image
import io
import base64
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pose import get_measurements, ellipse_circumference

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'front_image' not in request.files or 'side_image' not in request.files:
        return {'error': 'Both front and side images are required'}, 400
    
    front_image = request.files['front_image']
    side_image = request.files['side_image']
    height = request.form.get('height')

    if not height:
        return {'error': 'Height is required'}, 400

    try:
        height = float(height)
    except ValueError:
        return {'error': 'Invalid height value'}, 400

    if front_image.filename == '' or side_image.filename == '':
        return {'error': 'Both front and side images must be selected'}, 400

    # Save original files
    front_image_path = os.path.join(UPLOAD_FOLDER, 'front.jpg')
    side_image_path = os.path.join(UPLOAD_FOLDER, 'side.jpg')
    front_image.save(front_image_path)
    side_image.save(side_image_path)

    # Process images and compute measurements
    try:
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        detector = vision.PoseLandmarker.create_from_options(options)
        
        hips_span_front, waist_span_front, bust_span_front, biceps_span_front, arm_length_front, arm_span = get_measurements(front_image_path, detector)
        hips_span_side, waist_span_side, bust_span_side, biceps_span_side, arm_length_side, _ = get_measurements(side_image_path, detector)

        front_ratio = height * 0.9189 / arm_span
        hips_a = hips_span_front * front_ratio
        waist_a = waist_span_front * front_ratio
        bust_a = bust_span_front * front_ratio
        biceps_a = biceps_span_front * front_ratio

        side_ratio = front_ratio * arm_length_front / arm_length_side
        hips_b = hips_span_side * side_ratio
        waist_b = waist_span_side * side_ratio
        bust_b = bust_span_side * side_ratio
        biceps_b = biceps_span_side * side_ratio

        hips_circumference = ellipse_circumference(hips_a/2, hips_b/2)
        waist_circumference = ellipse_circumference(waist_a/2, waist_b/2)
        bust_circumference = ellipse_circumference(bust_a/2, bust_b/2)
        biceps_circumference = ellipse_circumference(biceps_a/2, biceps_b/2)

        measurements = {
            'hips_circumference': hips_circumference,
            'waist_circumference': waist_circumference,
            'bust_circumference': bust_circumference,
            'biceps_circumference': biceps_circumference
        }

        return {
            'message': 'Measurements computed successfully',
            'measurements': measurements
        }, 200
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)