# Gestus API - Binary Classifier Version

A Flask-based API for Brazilian Sign Language (Libras) recognition using binary transformer classifiers.

## Model Architecture

This API now uses **binary classifiers** instead of a single multiclass model:

- **One-vs-All Approach**: Each sign has its own dedicated binary classifier
- **BinaryTransformerClassifier**: Transformer-based architecture optimized for binary classification
- **Confidence Scoring**: Each model outputs a confidence score (0-1) for its specific sign

### Binary Models

Models are stored in the `./models/` directory.

Each `.pth` file contains a trained `BinaryTransformerClassifier` for that specific sign.

## API Endpoints

### `POST /process`

Process a video file and return sign predictions.

**Request:**

- `video`: Video file (WebM, MP4, etc.)

**Response:**

```json
{
  "status": "success",
  "predictions": [
    { "sign": "acontecer", "confidence": 0.8542 },
    { "sign": "filho", "confidence": 0.7231 },
    { "sign": "america", "confidence": 0.6891 }
  ]
}
```

### `GET /get-random-sign`

Get a sign that needs more training data.

### `POST /contribute`

Contribute a new video sample for training.

## Configuration

Update `config.py` to configure the models directory:

```python
# Models settings
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
```

## Testing

Run the test script to verify binary models:

```bash
python test_binary_models.py
```

This will:

1. Check if model files exist
2. Load all binary classifiers
3. Test inference on dummy data
4. Verify model architecture

## How It Works

1. **Video Processing**: Extract MediaPipe landmarks from video frames
2. **Preprocessing**: Normalize and sequence landmarks to fixed length (150 frames)
3. **Binary Inference**: Run input through all binary classifiers
4. **Confidence Ranking**: Sort predictions by confidence score
5. **Top-K Results**: Return top 3 most confident predictions

### Model Output

Each binary classifier outputs:

- Raw logit (unbounded value)
- Sigmoid probability (0-1 range)
- Higher values = more confident the sign is present

## Advantages of Binary Approach

1. **Scalability**: Easy to add new signs by training new binary models
2. **Interpretability**: Clear confidence scores for each sign
3. **Robustness**: One failing model doesn't affect others
4. **Performance**: Can parallelize inference across models
5. **Training Efficiency**: Easier to handle class imbalance per sign

## Requirements

See `requirements.txt` for dependencies. Key packages:

- `torch` - PyTorch for neural networks
- `mediapipe` - Landmark extraction
- `flask` - Web API framework
- `opencv-python` - Video processing
