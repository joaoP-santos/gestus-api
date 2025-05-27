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
from supabase import create_client, Client

# Import configuration from config.py
import config

from utils import initialize_model, process_video, MediaPipeLandmarkExtractor

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
    """Accepts a video file, runs inference, and returns predictions."""
    if "video" not in request.files:
        return jsonify({"status": "error", "error": "No video file provided"}), 400

    tmp_path = None
    video_file = request.files["video"]
    try:
        # Secure filename and determine suffix
        filename = secure_filename(video_file.filename)
        suffix = os.path.splitext(filename)[1] or ".webm"

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            video_file.save(tmp_path)

        # Run processing
        predictions = process_video(tmp_path, model, landmark_extractor)

        # Return results
        return jsonify({"status": "success", "predictions": predictions}), 200

    except ValueError as ve:
        # Validation-related error (e.g., hands not detected)
        return jsonify({"status": "error", "error": str(ve), "message": "Validation failed"}), 400

    except Exception as e:
        # Unexpected errors
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e), "message": "Server error occurred"}), 500

    finally:
        # Clean up temp file if it exists
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

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
    """Process a video for dataset contribution."""
    if "video" not in request.files:
        return jsonify({"status": "error", "error": "No video file provided"}), 400
    
    if "sign" not in request.form:
        return jsonify({"status": "error", "error": "No sign label provided"}), 400
    
    sign_name = request.form["sign"]
    if sign_name not in app.config['SIGNS']:
        return jsonify({"status": "error", "error": "Invalid sign name"}), 400
        
    tmp_path = None
    video_file = request.files["video"]
    try:
        # Secure filename and determine suffix
        filename = secure_filename(video_file.filename)
        suffix = os.path.splitext(filename)[1] or ".webm"

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            video_file.save(tmp_path)
            
        # Extract landmarks using our existing extractor
        landmarks_extractor = MediaPipeLandmarkExtractor()
        landmarks_sequence = []
        
        # Open video file
        import cv2
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open the video file: {tmp_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Handle invalid frame count (common in WebM files)
        if frame_count <= 0 or frame_count > 100000:
            frame_count = 300  # Default to processing up to 300 frames
        
        # Set frame dimensions for landmark extractor
        landmarks_extractor.set_frame_size(width, height)
        
        # Collect landmarks from each frame
        processed_frames = 0
        max_frames = min(300, frame_count)
        
        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                landmarks, _ = landmarks_extractor.extract_landmarks(frame)
                
                # Only add landmarks if they contain actual data (not empty or all zeros)
                if landmarks is not None and np.any(landmarks):
                    landmarks_sequence.append(landmarks.tolist())  # Convert numpy arrays to lists for JSON
                    
                processed_frames += 1
                
            except Exception as e:
                # Continue with next frame instead of failing completely
                continue
        
        cap.release()
        
        # Make sure we actually have landmarks before proceeding
        if not landmarks_sequence:
            raise ValueError("No landmarks could be extracted from the video. Please ensure your hands are clearly visible.")

        # Create a timestamp and unique ID for the file name
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8]  # First 8 chars of UUID
        
        # Create metadata
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
            storage_path = f"landmarks/{sign_name}/{filename}"
                
            # Upload the file to Supabase - use the simplified API that doesn't need bucket check
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
        
        except Exception as e:
            print(f"Error saving to Supabase: {e}")
        
        return jsonify({"status": "success", "message": "Video processed and saved successfully"}), 200

    except ValueError as ve:
        return jsonify({"status": "error", "error": str(ve), "message": "Validation failed"}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e), "message": "Server error occurred"}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

@app.route("/get-sample-video", methods=["GET"])
def get_sample_video():
    print(request)
    """Return a sample video URL for a given sign."""
    sign_name = request.args.get('sign')
    
    if not sign_name:
        return jsonify({"status": "error", "error": "Sign name is required"}), 400
    
    try:
        bucket_name = config.SUPABASE_BUCKET

        try:
            file_path = f"samples/{sign_name}.mp4"
            
            # Generate a signed URL for the video (valid for 1 hour)
            signed_url = supabase.storage.from_(bucket_name).create_signed_url(file_path, 3600)
            
            if signed_url.get('error'):
                return jsonify({"status": "error", "error": "Failed to generate video URL"}), 500
            
            return jsonify({
                "status": "success", 
                "video_url": signed_url['signedURL'],
            }), 200
            
        except Exception as storage_error:
            print(f"Storage error: {storage_error}")
            return jsonify({"status": "error", "error": "Failed to access sample videos"}), 500
            
    except Exception as e:
        print(f"Error fetching sample video: {e}")
        return jsonify({"status": "error", "error": "Server error occurred"}), 500

@app.route("/debug-storage", methods=["GET"])
def debug_storage():
    """Debug endpoint to see what's in Supabase storage."""
    try:
        bucket_name = config.SUPABASE_BUCKET
        
        # List root contents
        root_contents = supabase.storage.from_(bucket_name).list()
        
        # List samples folder if it exists
        samples_contents = []
        try:
            samples_contents = supabase.storage.from_(bucket_name).list("samples")
        except Exception as e:
            print(f"Error listing samples folder: {e}")
        
        return jsonify({
            "status": "success",
            "bucket": bucket_name,
            "root_contents": [{"name": f["name"], "type": f.get("metadata", {}).get("mimetype", "unknown")} for f in root_contents],
            "samples_contents": [{"name": f["name"], "type": f.get("metadata", {}).get("mimetype", "unknown")} for f in samples_contents]
        }), 200
        
    except Exception as e:
        print(f"Error in debug storage: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == "__main__":
    # Configuration from config.py
    port = config.PORT
    debug = config.FLASK_ENV == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
