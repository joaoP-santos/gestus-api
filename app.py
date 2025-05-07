import os
import tempfile
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import torch

from utils import initialize_model, process_video

app = Flask(__name__)
CORS(app)

# Load model at startup
# MODEL_PATH can be overridden via environment variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.environ.get("MODEL_PATH", "best_model.pth")
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

if __name__ == "__main__":
    # Bind to PORT ((env var provided by Render), default 5000
    port = int(os.environ.get("PORT", 5000))
    # Enable debug only in development
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
