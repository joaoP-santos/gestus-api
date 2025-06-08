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
try:
    supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    print("Supabase client initialized successfully")
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    # Create a dummy client to prevent crashes
    supabase = None

# Configuration from config.py
app.config['SIGNS'] = config.SIGNS

# Load model at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = config.MODEL_PATH
model, landmark_extractor = initialize_model(None, model_path, None)

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
    """Return the sign with the least number of samples for the user to record."""
    try:        # Count Supabase files only (no local file counting)
        supabase_sample_counts = {}
        try:
            if supabase is None:
                print("Warning: Supabase client not available for get-random-sign")
                raise Exception("Supabase client not available")
                
            bucket_name = config.SUPABASE_BUCKET
            
            # Try to list the landmarks folder to see what sign folders exist
            try:
                folders = supabase.storage.from_(bucket_name).list("landmarks")
                
                for folder in folders:
                    if folder['name'] in app.config['SIGNS']:
                        # List files in this sign folder
                        try:
                            files = supabase.storage.from_(bucket_name).list(f"landmarks/{folder['name']}")
                            json_files = [f for f in files if f['name'].endswith('.json')]
                            supabase_sample_counts[folder['name']] = len(json_files)
                        except:
                            # If folder doesn't exist or is empty, count as 0
                            supabase_sample_counts[folder['name']] = 0
            except:
                # If landmarks folder doesn't exist, check root level files as fallback
                try:
                    response = supabase.storage.from_(bucket_name).list()
                    
                    for file in response:
                        if file['name'].endswith('.json'):
                            # Extract sign name from filename format: sign_name_timestamp.json
                            sign_name = file['name'].split('_')[0]
                            if sign_name in app.config['SIGNS']:
                                supabase_sample_counts[sign_name] = supabase_sample_counts.get(sign_name, 0) + 1
                except:
                    # If everything fails, just continue with empty supabase counts
                    pass
        except Exception as e:
            print(f"Error fetching Supabase storage stats: {e}")
            # Don't fail the request if Supabase stats fail
        
        # Initialize all signs with 0 count if not already in supabase_sample_counts
        sign_counts = {}
        for sign in app.config['SIGNS']:
            sign_counts[sign] = supabase_sample_counts.get(sign, 0)
        
        # Find sign(s) with minimum count
        min_count = min(sign_counts.values()) if sign_counts else 0
        signs_with_min_count = [sign for sign, count in sign_counts.items() if count == min_count]
        
        # If multiple signs have the same minimum count, choose randomly among them
        selected_sign = random.choice(signs_with_min_count)
        
        return jsonify({
            "sign": selected_sign,
            "current_count": min_count,
            "all_counts": sign_counts
        })
        
    except Exception as e:
        # Fallback to random selection if there's any error
        print(f"Error in get_random_sign: {e}")
        random_sign = random.choice(app.config['SIGNS'])
        return jsonify({"sign": random_sign})

@app.route("/dataset-stats", methods=["GET"])
def get_dataset_stats():
    """Return statistics about the dataset from Supabase storage only."""
    try:
        print("Starting dataset stats request...")
        
        # Get stats from Supabase storage only
        supabase_stats = {
            "totalSupabaseSamples": 0,
            "supabaseSampleCounts": {}
        }        
        try:
            print("Attempting to connect to Supabase...")
            if supabase is None:
                print("Error: Supabase client is not initialized")
                raise Exception("Supabase client not available")
                
            bucket_name = config.SUPABASE_BUCKET
            print(f"Using bucket: {bucket_name}")
            
            # Try to list the landmarks folder to see what sign folders exist
            try:
                print("Listing landmarks folder...")
                folders = supabase.storage.from_(bucket_name).list("landmarks")
                print(f"Found {len(folders)} folders in landmarks")
                
                for folder in folders:
                    if folder['name'] in app.config['SIGNS']:
                        # List files in this sign folder
                        try:
                            files = supabase.storage.from_(bucket_name).list(f"landmarks/{folder['name']}")
                            json_files = [f for f in files if f['name'].endswith('.json')]
                            count = len(json_files)
                            supabase_stats["supabaseSampleCounts"][folder['name']] = count
                            supabase_stats["totalSupabaseSamples"] += count
                        except Exception as folder_e:
                            print(f"Error listing files in folder {folder['name']}: {folder_e}")
                            # If folder doesn't exist or is empty, count as 0
                            supabase_stats["supabaseSampleCounts"][folder['name']] = 0
            except Exception as landmarks_e:
                print(f"Landmarks folder error: {landmarks_e}")
                # If landmarks folder doesn't exist, check root level files as fallback
                try:
                    print("Trying root level files...")
                    response = supabase.storage.from_(bucket_name).list()
                    print(f"Found {len(response)} files in root")
                    
                    for file in response:
                        if file['name'].endswith('.json'):
                            supabase_stats["totalSupabaseSamples"] += 1
                            
                            # Extract sign name from filename format: sign_name_timestamp.json
                            sign_name = file['name'].split('_')[0]
                            if sign_name in app.config['SIGNS']:  # Ensure it's a valid sign
                                supabase_stats["supabaseSampleCounts"][sign_name] = supabase_stats["supabaseSampleCounts"].get(sign_name, 0) + 1
                except Exception as root_e:
                    print(f"Root level files error: {root_e}")
                    # If everything fails, just continue with empty supabase counts
                    pass
        except Exception as e:
            print(f"Error fetching Supabase storage stats: {e}")
            traceback.print_exc()
            # Don't fail the request if Supabase stats fail
        
        # Initialize all signs with 0 count if not in supabase counts
        all_sign_counts = {}
        for sign in app.config['SIGNS']:
            all_sign_counts[sign] = supabase_stats["supabaseSampleCounts"].get(sign, 0)
        
        print(f"Returning stats: {len(all_sign_counts)} signs, {supabase_stats['totalSupabaseSamples']} total samples")
        
        return jsonify({
            "supabaseStats": supabase_stats,
            "totalSamples": supabase_stats["totalSupabaseSamples"],
            "sampleCounts": all_sign_counts
        })
    except Exception as e:
        print(f"Critical error in dataset-stats: {e}")
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
            raise ValueError("No landmarks could be extracted from the video. Please ensure your hands are clearly visible.")        # Create metadata
        metadata = {
            "sign": sign_name,
            "landmarks": landmarks_sequence
        }
        
        # Create a timestamp and unique ID for the file name
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8]  # First 8 chars of UUID
        filename = f"{sign_name}_{timestamp}_{unique_id}.json"
        
        # Upload directly to Supabase storage (no local file creation)
        try:
            bucket_name = config.SUPABASE_BUCKET
            
            # Convert metadata to JSON string and encode as bytes
            json_data = json.dumps(metadata)
            file_content = json_data.encode('utf-8')
            
            # Use the sign name as the folder name
            storage_path = f"landmarks/{sign_name}/{filename}"
                
            # Upload the file to Supabase directly
            result = supabase.storage.from_(bucket_name).upload(
                path=storage_path, 
                file=file_content,
                file_options={"content-type": "application/json"}
            )
            print(f"Successfully uploaded {filename} to Supabase Storage bucket: {bucket_name} in folder: {sign_name}")
                    
        except Exception as e:
            print(f"Error saving to Supabase: {e}")
            # Return error if Supabase upload fails since we're not saving locally anymore
            return jsonify({"status": "error", "error": "Failed to save to cloud storage", "message": "Upload failed"}), 500
        
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

@app.route("/dataset-samples", methods=["GET"])
def get_dataset_samples():
    """Return dataset samples metadata (without landmark data for performance)."""
    try:
        samples = []
        
        try:
            bucket_name = config.SUPABASE_BUCKET
            
            # Try to list the landmarks folder to see what sign folders exist
            try:
                folders = supabase.storage.from_(bucket_name).list("landmarks")
                
                for folder in folders:
                    if folder['name'] in app.config['SIGNS']:
                        # List files in this sign folder
                        try:
                            files = supabase.storage.from_(bucket_name).list(f"landmarks/{folder['name']}")
                            json_files = [f for f in files if f['name'].endswith('.json')]                            # Limit to most recent 5 files per sign to reduce load
                            json_files = sorted(json_files, key=lambda x: x.get('updated_at', ''), reverse=True)[:5]
                            
                            for file in json_files:
                                try:
                                    # Extract timestamp from filename or use file metadata
                                    timestamp = file.get('updated_at') or file.get('created_at') or ''
                                    
                                    # Create sample entry WITHOUT landmarks for performance
                                    sample = {
                                        "id": file['name'].replace('.json', ''),
                                        "sign": folder['name'],
                                        "timestamp": timestamp,
                                        "metadata": {
                                            "fps": 30,  # Default value
                                            "duration": None,
                                            "width": None,
                                            "height": None
                                        }
                                    }
                                    samples.append(sample)
                                        
                                except Exception as e:
                                    print(f"Error processing file {file['name']}: {e}")
                                    continue
                                    
                        except Exception as e:
                            print(f"Error listing files in folder {folder['name']}: {e}")
                            continue
                            
            except Exception as e:
                print(f"Error listing landmarks folder: {e}")
                # If landmarks folder doesn't exist, check root level files as fallback
                try:
                    response = supabase.storage.from_(bucket_name).list()
                    json_files = [f for f in response if f['name'].endswith('.json')]
                      # Limit to most recent 50 files to avoid overwhelming the response
                    json_files = sorted(json_files, key=lambda x: x.get('updated_at', ''), reverse=True)[:50]
                    for file in json_files:
                        try:                            # Extract sign name from filename format: sign_name_timestamp.json
                            sign_name = file['name'].split('_')[0]
                            if sign_name not in app.config['SIGNS']:
                                continue
                                
                            timestamp = file.get('updated_at') or file.get('created_at') or ''
                            
                            # Create sample entry WITHOUT landmarks for performance
                            sample = {
                                "id": file['name'].replace('.json', ''),
                                "sign": sign_name,
                                "timestamp": timestamp,
                                "metadata": {
                                    "fps": 30,  # Default value
                                    "duration": None,
                                    "width": None,
                                    "height": None
                                }
                            }
                            samples.append(sample)
                                
                        except Exception as e:
                            print(f"Error processing root file {file['name']}: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Error listing root files: {e}")
                    
        except Exception as e:
            print(f"Error fetching Supabase samples: {e}")
        
        # Sort samples by timestamp (most recent first)
        samples = sorted(samples, key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            "samples": samples,
            "total_samples": len(samples)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error getting dataset samples: {str(e)}"}), 500

@app.route("/sample-landmarks/<sample_id>", methods=["GET"])
def get_sample_landmarks(sample_id):
    """Return landmarks for a specific sample."""
    try:
        bucket_name = config.SUPABASE_BUCKET
        
        # Try to find the sample file by searching through the landmarks folder structure
        try:
            # First check organized landmarks folder structure
            folders = supabase.storage.from_(bucket_name).list("landmarks")
            
            for folder in folders:
                if folder['name'] in app.config['SIGNS']:
                    try:
                        files = supabase.storage.from_(bucket_name).list(f"landmarks/{folder['name']}")
                        
                        for file in files:
                            if file['name'].replace('.json', '') == sample_id:
                                file_path = f"landmarks/{folder['name']}/{file['name']}"
                                response = supabase.storage.from_(bucket_name).download(file_path)
                                
                                if response:
                                    import json
                                    landmark_data = json.loads(response.decode('utf-8'))
                                    
                                    return jsonify({
                                        "landmarks": landmark_data.get('landmarks', []),
                                        "metadata": {
                                            "fps": landmark_data.get('fps', 30),
                                            "frame_count": len(landmark_data.get('landmarks', [])),
                                            "duration": landmark_data.get('duration'),
                                            "width": landmark_data.get('width'),
                                            "height": landmark_data.get('height')
                                        }
                                    })
                                    
                    except Exception as e:
                        print(f"Error searching in folder {folder['name']}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error searching landmarks folder: {e}")
            
        # Fallback: check root level files
        try:
            response = supabase.storage.from_(bucket_name).list()
            json_files = [f for f in response if f['name'].endswith('.json')]
            
            for file in json_files:
                if file['name'].replace('.json', '') == sample_id:
                    response = supabase.storage.from_(bucket_name).download(file['name'])
                    
                    if response:
                        import json
                        landmark_data = json.loads(response.decode('utf-8'))
                        
                        return jsonify({
                            "landmarks": landmark_data.get('landmarks', []),
                            "metadata": {
                                "fps": landmark_data.get('fps', 30),
                                "frame_count": len(landmark_data.get('landmarks', [])),
                                "duration": landmark_data.get('duration'),
                                "width": landmark_data.get('width'),
                                "height": landmark_data.get('height')
                            }
                        })
                        
        except Exception as e:
            print(f"Error searching root files: {e}")
            
        return jsonify({"error": "Sample not found"}), 404
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error getting sample landmarks: {str(e)}"}), 500

if __name__ == "__main__":
    print("Starting Gestus API...")
    print(f"Model path: {config.MODEL_PATH}")
    print(f"Supabase URL: {config.SUPABASE_URL}")
    print(f"Available signs: {len(config.SIGNS)}")
    print(f"Port: {config.PORT}")
    
    # Test Supabase connection
    try:
        print("Testing Supabase connection...")
        bucket_name = config.SUPABASE_BUCKET
        supabase.storage.from_(bucket_name).list()
        print("Supabase connection successful!")
    except Exception as e:
        print(f"Warning: Supabase connection failed: {e}")
        print("The API will continue but dataset stats may not work properly.")
    
    # Configuration from config.py
    port = config.PORT
    debug = config.FLASK_ENV == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
