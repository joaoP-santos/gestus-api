from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import torch
import tempfile
import os
import traceback
from werkzeug.utils import secure_filename
import torch.nn.functional as F
import json
from flask_cors import CORS



from utils import process_video, initialize_model

app = Flask(__name__)
CORS(app)

# Initialize model and landmark extractor globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'best_model.pth'

# Initialize these as None first, then load in a function to handle errors better
model = None
idx_to_class = None
landmark_extractor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    
    # Save uploaded video to temporary file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            tmp_path = tmp.name
            video_file.save(tmp_path)
        
        # Add debug info
        print(f"Video saved to temporary file: {tmp_path}")
        print(f"Model loaded: {model is not None}")
        print(f"Class mapping size: {len(idx_to_class) if idx_to_class else 0}")
        
        # Process the video and get predictions
        print("\n--- Starting new video processing ---")
        predictions = process_video(tmp_path, model, landmark_extractor)
        
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
        return jsonify({
            'status': 'success',
            'predictions': predictions
        })
        
    except ValueError as ve:
        # Specific error for validation issues
        print(f"Validation error: {ve}")
        
        # Clean up on error
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
        # Return user-friendly error message
        return jsonify({
            'status': 'error',
            'error': str(ve),
            'message': 'Please try recording again with clearer gestures and make sure your hands are visible.'
        }), 400  # 400 Bad Request
        
    except Exception as e:
        # Print full traceback for debugging
        print(f"Unexpected error processing video: {e}")
        traceback.print_exc()
        
        # Clean up on error
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
        # Return detailed error message
        error_message = str(e)
        return jsonify({
            'status': 'error',
            'error': error_message,
            'message': 'A server error occurred while processing your video. Our team has been notified of the issue.'
        }), 500

# Initialize model at startup
model, landmark_extractor = initialize_model(model, model_path, landmark_extractor)

if __name__ == '__main__':
    app.run(debug=True)
