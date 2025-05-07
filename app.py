import os
import tempfile
import traceback
import json
import uuid
import datetime
import random
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
from supabase import create_client, Client

# Import configuration from config.py
import config

# Import utilities for landmark processing and model handling
from utils import initialize_model, process_raw_landmarks, MediaPipeLandmarkExtractor
# Import the utils module to access the global idx_to_class variable
import utils

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize Supabase client
supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

# Configuration from config.py
app.config['SIGNS'] = config.SIGNS

# Load model at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = config.MODEL_PATH
model, landmark_extractor = initialize_model(None, model_path, None)

# Dataset storage setup
os.makedirs('dataset', exist_ok=True)

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200

@app.route("/process", methods=["POST"])
def process_endpoint():
    """Accepts pre-processed landmarks from the frontend, runs inference, and returns predictions."""
    try:
        # Get JSON data from the request
        request_data = request.get_json()
        
        if not request_data or 'landmarks' not in request_data:
            return jsonify({"status": "error", "error": "No landmarks provided"}), 400
            
        # Extract landmarks sequence from the request
        landmarks_sequence = np.array(request_data['landmarks'])
        
        # Validate landmark data
        if landmarks_sequence.size == 0:
            return jsonify({"status": "error", "error": "Empty landmarks array provided"}), 400
        
        # Process the landmarks and get model predictions
        try:
            # Process landmarks to normalize sequence length
            processed_landmarks = process_raw_landmarks(
                landmarks_sequence,
                target_length=150,
                downsample_factor=2
            )
            
            # Get predictions from model
            with torch.no_grad():
                input_tensor = torch.tensor(processed_landmarks, dtype=torch.float32).unsqueeze(0).to(device)
                
                output = model(input_tensor)
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(output, dim=1)[0]
                
                # Get top predictions
                k = min(3, len(probabilities))
                topk_probs, topk_indices = torch.topk(probabilities, k)
                
                # Format predictions for output
                predictions = []
                for idx, prob in zip(topk_indices, topk_probs):
                    idx_val = idx.item()
                    
                    # Try multiple formats of the index to find the class
                    if idx_val in utils.idx_to_class:
                        sign = utils.idx_to_class[idx_val]
                    elif str(idx_val) in utils.idx_to_class:
                        sign = utils.idx_to_class[str(idx_val)]
                    else:
                        # If not found, use a placeholder
                        sign = f"Class_{idx_val}"
                    
                    predictions.append({
                        "sign": sign,
                        "confidence": float(prob.item())
                    })
                
                return jsonify({"status": "success", "predictions": predictions}), 200
                
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(e), "message": "Error processing landmarks"}), 500

    except ValueError as ve:
        # Validation-related error
        return jsonify({"status": "error", "error": str(ve), "message": "Validation failed"}), 400

    except Exception as e:
        # Unexpected errors
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e), "message": "Server error occurred"}), 500

@app.route("/get-random-sign", methods=["GET"])
def get_random_sign():
    """Return a random sign for the user to record."""
    random_sign = random.choice(app.config['SIGNS'])
    return jsonify({"sign": random_sign})

@app.route("/dataset-stats", methods=["GET"])
def get_dataset_stats():
    """Return statistics about the dataset."""
    try:
        # Count files in the dataset directory
        dataset_path = "dataset"
        total_samples = 0
        sample_counts = {}
        
        # Check if the directory exists
        if os.path.exists(dataset_path):
            for filename in os.listdir(dataset_path):
                if filename.endswith('.json'):
                    total_samples += 1
                    
                    # Extract sign name from filename format: sign_name_timestamp.json
                    sign_name = filename.split('_')[0]
                    sample_counts[sign_name] = sample_counts.get(sign_name, 0) + 1
        
        # Also get stats from Supabase if credentials are available
        supabase_stats = {
            "totalSupabaseSamples": 0,
            "supabaseSampleCounts": {}
        }
        
        try:
            # List files from Supabase storage
            bucket_name = config.SUPABASE_BUCKET
            response = supabase.storage.from_(bucket_name).list()
            
            for file in response:
                if file['name'].endswith('.json'):
                    supabase_stats["totalSupabaseSamples"] += 1
                    
                    # Extract sign name from filename format: sign_name_timestamp.json
                    sign_name = file['name'].split('_')[0]
                    if sign_name in app.config['SIGNS']:  # Ensure it's a valid sign
                        supabase_stats["supabaseSampleCounts"][sign_name] = supabase_stats["supabaseSampleCounts"].get(sign_name, 0) + 1
        except Exception as e:
            print(f"Error fetching Supabase storage stats: {e}")
            # Don't fail the request if Supabase stats fail
        
        return jsonify({
            "localStats": {
                "totalSamples": total_samples,
                "sampleCounts": sample_counts
            },
            "supabaseStats": supabase_stats,
            "totalSamples": total_samples + supabase_stats["totalSupabaseSamples"],
            "combinedCounts": {
                sign: sample_counts.get(sign, 0) + supabase_stats["supabaseSampleCounts"].get(sign, 0)
                for sign in set(list(sample_counts.keys()) + list(supabase_stats["supabaseSampleCounts"].keys()))
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error getting dataset stats: {str(e)}"}), 500

@app.route("/contribute", methods=["POST"])
def contribute_endpoint():
    """Accept pre-processed landmarks for dataset contribution."""
    try:
        # Get JSON data from the request
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({"status": "error", "error": "No data provided"}), 400
            
        # Validate required fields
        if 'sign' not in request_data:
            return jsonify({"status": "error", "error": "No sign label provided"}), 400
            
        if 'landmarks' not in request_data:
            return jsonify({"status": "error", "error": "No landmarks provided"}), 400
        
        # Extract data
        sign_name = request_data['sign']
        landmarks_sequence = request_data['landmarks']
        
        # Validate sign name
        if sign_name not in app.config['SIGNS']:
            return jsonify({"status": "error", "error": "Invalid sign name"}), 400
            
        # Validate landmarks data
        if not landmarks_sequence or len(landmarks_sequence) == 0:
            return jsonify({"status": "error", "error": "Empty landmarks array provided"}), 400

        # Create a timestamp and unique ID for the file name
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8]  # First 8 chars of UUID
        
        # Create metadata - only include sign and landmarks fields
        metadata = {
            "sign": sign_name,
            "landmarks": landmarks_sequence
        }
        
        # Save to local file
        filename = f"{sign_name}_{timestamp}_{unique_id}.json"
        filepath = os.path.join("dataset", filename)
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f)
            
        # Upload to Supabase storage
        try:
            bucket_name = config.SUPABASE_BUCKET
            
            # Directly try to upload to the existing bucket
            with open(filepath, 'rb') as f:
                file_content = f.read()
            
            # Use the sign name as the folder name
            storage_path = f"{sign_name}/{filename}"
                
            # Upload the file to Supabase
            try:
                result = supabase.storage.from_(bucket_name).upload(
                    path=storage_path, 
                    file=file_content,
                    file_options={"content-type": "application/json"}
                )
                print(f"Successfully uploaded {filename} to Supabase Storage bucket: {bucket_name} in folder: {sign_name}")
                
                # Delete the local file after successful upload to Supabase
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"Deleted local file: {filepath}")
                    
            except Exception as upload_error:
                print(f"Error uploading {filename} to Supabase Storage: {upload_error}")
                raise upload_error
        
        except Exception as e:
            print(f"Error saving to Supabase: {e}")
            # Don't fail the request if Supabase upload fails
            # but keep the error information
            return jsonify({
                "status": "partial_success", 
                "message": "Contribution saved locally but not to cloud storage",
                "error": str(e)
            }), 207
        
        return jsonify({
            "status": "success",
            "message": "Contribution received successfully",
            "sign": sign_name,
            "id": unique_id
        }), 200

    except ValueError as ve:
        return jsonify({"status": "error", "error": str(ve), "message": "Validation failed"}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e), "message": "Server error occurred"}), 500

if __name__ == "__main__":
    # Configuration from config.py
    port = config.PORT
    debug = config.FLASK_ENV == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
