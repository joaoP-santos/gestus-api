import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import traceback
import json
import mediapipe as mp
from scipy.interpolate import interp1d

# Import your existing recognition code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MediaPipeLandmarkExtractor:
    """Extract and visualize hand and pose landmarks using MediaPipe"""
    
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize models with good params for real-time
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  # Use lite model for better speed
        )
        
        # Add camera settings
        self.frame_size = (640, 480)  # Default size
        self.prev_frame_time = 0
        self.fps = 0
    
    def set_frame_size(self, width, height):
        """Update frame size settings"""
        self.frame_size = (width, height)
    
    def extract_landmarks(self, frame):
        """Extract landmarks from a frame using MediaPipe"""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Process frame with both models - use image dimensions to fix the NORM_RECT error
        hands_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)
        
        # Extract landmarks into a flat array - MUST MATCH DATASET EXTRACTION ORDER
        landmarks = []
        
        # Initialize a list to hold all landmarks with padding
        num_pose_landmarks = 33
        num_hand_landmarks = 21
        
        # Define the expected number of landmarks for each type
        all_landmarks = np.zeros(3 * (num_pose_landmarks + 2 * num_hand_landmarks))
        
        offset = 0
        
        # 1. FIRST extract pose landmarks (matching the dataset extraction order)
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                all_landmarks[offset:offset + 3] = [landmark.x, landmark.y, landmark.z]
                offset += 3
        else:
            # Add zeros for missing pose
            offset += num_pose_landmarks * 3
        
        # 2. THEN extract left hand landmarks
        left_hand_landmarks = None
        right_hand_landmarks = None
        
        # Identify left/right hands
        if hands_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                if len(hands_results.multi_handedness) > hand_idx and \
                  len(hands_results.multi_handedness[hand_idx].classification) > 0:
                    if hands_results.multi_handedness[hand_idx].classification[0].label == 'Left':
                        left_hand_landmarks = hand_landmarks
                    else:
                        right_hand_landmarks = hand_landmarks
        
        # Extract left hand landmarks
        if left_hand_landmarks:
            for landmark in left_hand_landmarks.landmark:
                all_landmarks[offset:offset + 3] = [landmark.x, landmark.y, landmark.z]
                offset += 3
        else:
            # Add zeros for missing left hand
            offset += num_hand_landmarks * 3
        
        # 3. FINALLY extract right hand landmarks
        if right_hand_landmarks:
            for landmark in right_hand_landmarks.landmark:
                all_landmarks[offset:offset + 3] = [landmark.x, landmark.y, landmark.z]
                offset += 3
        else:
            # Add zeros for missing right hand
            offset += num_hand_landmarks * 3
        
        # Bundle results for drawing
        results = (hands_results, pose_results, left_hand_landmarks is not None or right_hand_landmarks is not None)
        
        return all_landmarks, results
    
    def draw_landmarks(self, frame, results):
        """Draw landmarks on the frame"""
        hands_results, pose_results, _ = results
        
        # Draw hand landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            # Only draw upper body landmarks (more relevant for sign language)
            upper_body_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
            connections = [conn for conn in self.mp_pose.POSE_CONNECTIONS 
                          if conn[0] in upper_body_indices and conn[1] in upper_body_indices]
            
            # Draw the filtered landmarks and connections
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                connections,
                self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return frame

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, num_heads=4, dropout=0.3):
        super(TransformerClassifier, self).__init__()
        
        # Input feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size)
        
        # Transformer encoder (multi-head attention + feed forward)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads,
            dim_feedforward=hidden_size*2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Global pooling with attention
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        """Forward pass with better error handling"""
        try:
            # Now all sequences should have the same length, so we can
            # skip the padding mask since it's not needed
            
            # x: [batch_size, seq_len, input_size]
            batch_size, seq_len, _ = x.shape
            
            # Project input to hidden dimension
            features = self.feature_projection(x)  # [batch_size, seq_len, hidden_size]
            
            # Add positional encoding
            features = self.positional_encoding(features)
            
            # Pass through transformer encoder without mask
            transformer_output = self.transformer_encoder(features)
            
            # Apply attention pooling
            attention_weights = self.attention_pooling(transformer_output)
            weighted_output = attention_weights * transformer_output
            
            # Sum along sequence dimension
            pooled = weighted_output.sum(dim=1)
            
            # Classify
            logits = self.classifier(pooled)
            
            return logits
        except RuntimeError as e:
            print(f"Error in transformer forward pass: {e}")
            # Re-raise the exception for proper debugging
            raise

def normalize_sequence_length(landmarks, target_length=150):
    """
    Stretch or squeeze landmark sequences to a fixed length using interpolation.
    This preserves the temporal pattern better than padding.
    
    Args:
        landmarks: numpy array of shape [frames, features]
        target_length: desired sequence length
        
    Returns:
        numpy array of shape [target_length, features]
    """
    # If sequence is empty, return zeros
    if landmarks.shape[0] == 0:
        return np.zeros((target_length, landmarks.shape[1]))
    
    # If sequence is already the target length, return as is
    if landmarks.shape[0] == target_length:
        return landmarks
        
    # Create time points for original sequence
    original_times = np.linspace(0, 1, landmarks.shape[0])
    
    # Create time points for target sequence
    target_times = np.linspace(0, 1, target_length)
    
    # Create interpolation function for each feature
    normalized_landmarks = np.zeros((target_length, landmarks.shape[1]))
    
    # Interpolate each feature separately
    for i in range(landmarks.shape[1]):
        # Handle case where original sequence is length 1
        if landmarks.shape[0] == 1:
            # Just repeat the single frame
            normalized_landmarks[:, i] = landmarks[0, i]
        else:
            # Create interpolation function (cubic if enough points, otherwise linear)
            kind = 'cubic' if landmarks.shape[0] > 3 else 'linear'
            interpolator = interp1d(
                original_times, landmarks[:, i], 
                kind=kind, 
                bounds_error=False,  # Don't raise error for out-of-bounds
                fill_value=(landmarks[0, i], landmarks[-1, i])  # Use endpoints for out-of-bounds
            )
            
            # Interpolate to target sequence length
            normalized_landmarks[:, i] = interpolator(target_times)
    
    return normalized_landmarks


def process_raw_landmarks(landmarks, target_length, downsample_factor):
    """Process landmarks without normalization for single sign recognition"""
    # Replace any NaN or Inf values
    landmarks = np.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Downsample to save memory and computation
    landmarks = landmarks[::downsample_factor]
    
    # Use interpolation to normalize sequence length
    normalized_landmarks = normalize_sequence_length(landmarks, target_length)
    
    return normalized_landmarks

def process_video(video_path, model, landmark_extractor):
    """Process a video file to extract landmarks and recognize signs"""

    # Open video file
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open the video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle invalid frame count (common in WebM files)
    if frame_count <= 0 or frame_count > 100000:  # Unreasonable frame count
        print(f"Invalid frame count detected: {frame_count}, setting to default max")
        frame_count = 300  # Default to processing up to 300 frames
    
    print(f"Video properties: {width}x{height}, {fps} fps, using up to {frame_count} frames")
    
    # Set frame dimensions for landmark extractor
    landmark_extractor.set_frame_size(width, height)
    
    # Collect landmarks from each frame
    landmarks_sequence = []
    processed_frames = 0
    max_frames = min(300, frame_count)  # Limit processing to 300 frames (~10 seconds at 30fps)
    
    print(f"Processing up to {max_frames} frames")
    while processed_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract landmarks from frame - fixing the MediaPipe error by providing dimensions
        try:
            landmarks, results = landmark_extractor.extract_landmarks(frame)
            
            # Only add landmarks if they contain actual data (not empty or all zeros)
            if landmarks is not None and np.any(landmarks):
                landmarks_sequence.append(landmarks)
                
            processed_frames += 1
            
            # Log progress every 30 frames
            if processed_frames % 30 == 0:
                print(f"Processed {processed_frames}/{max_frames} frames")
                
        except Exception as e:
            print(f"Error extracting landmarks from frame {processed_frames}: {e}")
            # Continue with next frame instead of failing completely
            continue
    
    cap.release()
    
    print(f"Extracted landmarks from {len(landmarks_sequence)} frames")
    
    # Make sure we actually have landmarks before proceeding
    if not landmarks_sequence:
        raise ValueError("No landmarks could be extracted from the video. Please ensure your hands are clearly visible.")
    
    # FIX: Match the training pipeline exactly
    # Convert to numpy array first as in the training code
    landmarks_array = np.array(landmarks_sequence)
    print(f"Raw landmarks shape: {landmarks_array.shape}")
    
    # Check for NaN or inf values
    if np.isnan(landmarks_array).any() or np.isinf(landmarks_array).any():
        print("WARNING: NaN or Inf values found in landmarks, replacing with zeros")
        landmarks_array = np.nan_to_num(landmarks_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Use the training pipeline's process_raw_landmarks function
    try:
        processed_landmarks = process_raw_landmarks(
            landmarks_array,
            target_length=150,
            downsample_factor=2
        )
        print(f"Processed landmarks shape: {processed_landmarks.shape}")
    except Exception as e:
        print(f"Error during landmark processing: {e}")
        traceback.print_exc()
        raise
    
    # Get predictions from model
    try:
        with torch.no_grad():
            input_tensor = torch.tensor(processed_landmarks, dtype=torch.float32).unsqueeze(0).to(device)
            print(f"Input tensor shape: {input_tensor.shape}")
            
            output = model(input_tensor)
            print(f"Model output shape: {output.shape}")
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=1)[0]
            print(f"Probabilities tensor shape: {probabilities.shape}")
            print(f"Softmax probabilities sum: {probabilities.sum().item():.4f}")
            
            # Get top predictions
            k = min(3, len(probabilities))
            topk_probs, topk_indices = torch.topk(probabilities, k)
            
            # Debug: Print raw predictions
            print(f"Top indices: {topk_indices}")
            print(f"Top probs: {topk_probs}")
            
            # Format predictions for output
            predictions = []
            for idx, prob in zip(topk_indices, topk_probs):
                idx_val = idx.item()
                
                # Try multiple formats of the index to find the class
                if idx_val in idx_to_class:
                    sign = idx_to_class[idx_val]
                elif str(idx_val) in idx_to_class:
                    sign = idx_to_class[str(idx_val)]
                else:
                    # If not found, use a placeholder
                    sign = f"Class_{idx_val}"
                    print(f"WARNING: Index {idx_val} not found in class mapping!")
                    
                predictions.append({
                    "sign": sign,
                    "confidence": float(prob.item())
                })
            
            # Debug: Print mapped predictions
            print(f"Mapped predictions: {predictions}")
            
            return predictions
    except Exception as e:
        print(f"Error during inference: {e}")
        traceback.print_exc()
        raise

def initialize_model(model, model_path, landmark_extractor):
    """Initialize model separately to handle errors better"""
    global idx_to_class
    
    if model is None:
        print(f"Loading model from {model_path}")
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"ERROR: Model file not found at {model_path}")
                
                # Try to find model in alternative locations
                alt_paths = [
                    './best_model.pth',
                    './checkpoints/best_model.pth',
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_model.pth')
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        print(f"Found model at alternative path: {alt_path}")
                        model_path = alt_path
                        break
                        
                if not os.path.exists(model_path):
                    print("Could not find model file in any location.")
                    return False
            
            # Direct loading from checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            print(f"Model loaded from {model_path} successfully")
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Get model parameters and class mapping
            class_mapping = checkpoint.get('class_mapping', {})
            
            # Debug information about the model
            print(f"Model state dict keys: {list(checkpoint['model_state_dict'].keys())[:5]}...")
            
            if not class_mapping:
                print("WARNING: No class mapping found in checkpoint!")
                
                # Try looking for mapping file in same directory
                mapping_paths = [
                    os.path.join(os.path.dirname(model_path), 'class_mapping.json'),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'class_mapping.json'),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'class_mapping.json')
                ]
                
                for mapping_path in mapping_paths:
                    if os.path.exists(mapping_path):
                        try:
                            with open(mapping_path, 'r') as f:
                                class_mapping = json.load(f)
                            print(f"Loaded class mapping from {mapping_path}")
                            break
                        except Exception as e:
                            print(f"Error loading class mapping from {mapping_path}: {e}")
                
                # If still no mapping, create default
                if not class_mapping:
                    num_classes = checkpoint.get('num_classes', 0)
                    if num_classes > 0:
                        print(f"Creating default class mapping for {num_classes} classes")
                        class_mapping = {str(i): f"Class_{i}" for i in range(num_classes)}
            
            print(f"Class mapping has {len(class_mapping)} entries")
            print(f"Sample class mapping entries: {list(class_mapping.items())[:3]}...")
            
            # Get architecture parameters with defaults
            input_size = checkpoint.get('input_size', 225)
            hidden_size = checkpoint.get('hidden_size', 128)
            num_classes = checkpoint.get('num_classes', len(class_mapping))
            num_layers = checkpoint.get('num_layers', 2)
            num_heads = checkpoint.get('num_heads', 4)
            dropout = checkpoint.get('dropout', 0.3)
            
            print(f"Creating model with: input_size={input_size}, hidden_size={hidden_size}, "
                  f"num_classes={num_classes}, num_layers={num_layers}, num_heads={num_heads}")
            
            # Create the model
            model = TransformerClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_classes=num_classes,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout
            )
            
            # Load weights
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Model state_dict loaded successfully")
            except Exception as e:
                print(f"Error loading state_dict: {e}")
                return False
                
            model.to(device)
            model.eval()
            
            # Create mapping for predictions - handle both string and int keys
            idx_to_class = {}
            for class_name, idx in class_mapping.items():
                # Handle both string and int formats
                if isinstance(idx, int):
                    idx_to_class[idx] = class_name
                idx_to_class[str(idx)] = class_name
                
            print(f"Model loaded successfully with {len(idx_to_class)} classes")
            
            # Run a test prediction to verify model works
            try:
                test_input = torch.zeros((1, 150, input_size), device=device)
                with torch.no_grad():
                    output = model(test_input)
                    probs = F.softmax(output, dim=1)[0]
                    top_prob, top_idx = torch.max(probs, 0)
                    print(f"Test prediction: class {top_idx.item()} with prob {top_prob.item():.4f}")
                    
                    if top_idx.item() in idx_to_class:
                        print(f"Class name: {idx_to_class[top_idx.item()]}")
                    elif str(top_idx.item()) in idx_to_class:
                        print(f"Class name: {idx_to_class[str(top_idx.item())]}")
                    else:
                        print(f"Class index {top_idx.item()} not found in mapping!")
            except Exception as e:
                print(f"Error during test prediction: {e}")
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return False
            
    if landmark_extractor is None:
        try:
            landmark_extractor = MediaPipeLandmarkExtractor()
            print("MediaPipe landmark extractor initialized")
        except Exception as e:
            print(f"Error initializing landmark extractor: {e}")
            traceback.print_exc()
            return False
        
    return model, landmark_extractor
