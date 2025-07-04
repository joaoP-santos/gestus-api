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

# Global variables for binary models
binary_models = {}
idx_to_class = {}

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

# --- Binary Classifier for One-vs-All ---
class BinaryTransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, num_heads=4, dropout=0.3):
        super(BinaryTransformerClassifier, self).__init__()
        
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
        
        # Binary classification head (output 1 value for binary decision)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  # Binary classification: 1 output
        )
        
    def forward(self, x):
        """Forward pass for binary classification"""
        try:
            # x: [batch_size, seq_len, input_size]
            batch_size, seq_len, _ = x.shape
            
            # Project input to hidden dimension
            features = self.feature_projection(x)  # [batch_size, seq_len, hidden_size]
            
            # Add positional encoding
            features = self.positional_encoding(features)
            
            # Pass through transformer encoder
            transformer_output = self.transformer_encoder(features)
            
            # Apply attention pooling
            attention_weights = self.attention_pooling(transformer_output)
            weighted_output = attention_weights * transformer_output
            
            # Sum along sequence dimension
            pooled = weighted_output.sum(dim=1)
            
            # Binary classification output
            logits = self.classifier(pooled)
            
            return logits
        except RuntimeError as e:
            print(f"Error in binary transformer forward pass: {e}")
            raise

def temporal_smooth(signal, window_size=5):
    """
    Apply temporal smoothing to a signal using a moving average with edge handling.
    
    Args:
        signal: 1D numpy array, the signal to smooth
        window_size: size of the smoothing window (odd number recommended)
        
    Returns:
        Smoothed signal of same length
    """
    if len(signal) < window_size:
        return signal
        
    smoothed = np.copy(signal)
    half_window = window_size // 2
    
    # Handle the central part with full window
    for i in range(half_window, len(signal) - half_window):
        smoothed[i] = np.mean(signal[i-half_window:i+half_window+1])
    
    # Handle the edges with smaller windows
    for i in range(half_window):
        # Left edge
        smoothed[i] = np.mean(signal[:i+half_window+1])
        # Right edge
        smoothed[-(i+1)] = np.mean(signal[-(i+half_window+1):])
        
    return smoothed

def normalize_landmarks_spatially(landmarks, landmark_connections=None):
    """
    Robust spatial normalization of landmarks to make them invariant to position, scale, and orientation.
    Includes temporal smoothing and outlier protection to prevent normalization spikes.
    
    Args:
        landmarks: numpy array of shape [frames, num_landmarks*3] 
                  or [frames, num_landmarks, 3]
        landmark_connections: Optional dict mapping landmark indices to their connections
                            for more advanced normalization (if None, uses default)
                            
    Returns:
        numpy array with same shape as input but normalized
    """
    if landmarks.shape[0] == 0:
        return landmarks
    
    # Reshape to [frames, landmarks, 3] if necessary
    original_shape = landmarks.shape
    if len(original_shape) == 2 and original_shape[1] % 3 == 0:
        num_landmarks_in_data = original_shape[1] // 3
        landmarks = landmarks.reshape(original_shape[0], num_landmarks_in_data, 3)
    elif len(original_shape) == 3:
        num_landmarks_in_data = original_shape[1] # Already in [frames, num_landmarks, 3]
    else:
        # Cannot determine landmark structure or not 2D/3D array
        return landmarks # Return as is if shape is unexpected
    
    normalized = landmarks.copy()
    num_frames, num_landmarks_actual, _ = normalized.shape
    
    if num_landmarks_actual == 0: # No actual landmarks to process
        if len(original_shape) == 2:
             return np.zeros(original_shape) # return zeros of original shape
        return landmarks # Or return as is if it was 3D but with 0 landmarks

    reference_points = np.zeros((num_frames, 3))
    scale_factors = np.zeros(num_frames)
    
    # Landmark indices based on MediaPipe's typical combined output if all are present
    # Pose (33 landmarks), Left Hand (21 landmarks), Right Hand (21 landmarks)
    # Total = 33 + 21 + 21 = 75 landmarks.
    # Indices for pose (if present):
    pose_shoulder_l_idx, pose_shoulder_r_idx = 11, 12
    pose_hip_l_idx, pose_hip_r_idx = 23, 24
    # Indices for hands (relative to their own blocks, or absolute if concatenated):
    # If concatenated: Left Hand wrist = 33, Right Hand wrist = 33 + 21 = 54
    
    for frame_idx in range(num_frames):
        frame_landmarks = normalized[frame_idx]
        current_ref_landmarks = []
        
        # Try to use pose landmarks for reference if they seem to be present
        # Assuming pose landmarks are the first 33*3 features if present
        has_pose_data = num_landmarks_actual >= 33 
        
        if has_pose_data:
            # Check if shoulder/hip landmarks are non-zero
            shoulders = [frame_landmarks[pose_shoulder_l_idx], frame_landmarks[pose_shoulder_r_idx]]
            hips = [frame_landmarks[pose_hip_l_idx], frame_landmarks[pose_hip_r_idx]]
            
            valid_shoulders = [lm for lm in shoulders if not np.all(lm == 0)]
            valid_hips = [lm for lm in hips if not np.all(lm == 0)]

            current_ref_landmarks.extend(valid_shoulders)
            current_ref_landmarks.extend(valid_hips)

            if len(valid_shoulders) == 2:
                scale_dist = np.linalg.norm(valid_shoulders[0] - valid_shoulders[1])
                scale_factors[frame_idx] = scale_dist if scale_dist > 1e-6 else 1.0
            elif len(valid_hips) == 2:
                scale_dist = np.linalg.norm(valid_hips[0] - valid_hips[1])
                scale_factors[frame_idx] = scale_dist if scale_dist > 1e-6 else 1.0
            else: # Fallback scale if primary refs are not good
                scale_factors[frame_idx] = 1.0
        
        # If not enough pose data for reference, or if pose is not dominant, consider hands
        # This logic might need to be more sophisticated based on expected input
        if not current_ref_landmarks: # or some other condition to prefer hands
            # Assuming hands are after pose, or are the only landmarks
            # If only hands (21 left + 21 right = 42 landmarks total)
            # Left hand wrist: 0, Right hand wrist: 21
            # If pose + hands (33 pose + 21 left + 21 right = 75 landmarks total)
            # Left hand wrist: 33, Right hand wrist: 33 + 21 = 54
            
            left_hand_wrist_idx = 0 if num_landmarks_actual == 42 else (33 if num_landmarks_actual == 75 else -1)
            right_hand_wrist_idx = 21 if num_landmarks_actual == 42 else (54 if num_landmarks_actual == 75 else -1)

            hand_refs_to_check = []
            if left_hand_wrist_idx != -1 and not np.all(frame_landmarks[left_hand_wrist_idx] == 0):
                hand_refs_to_check.append(frame_landmarks[left_hand_wrist_idx])
            if right_hand_wrist_idx != -1 and not np.all(frame_landmarks[right_hand_wrist_idx] == 0):
                 hand_refs_to_check.append(frame_landmarks[right_hand_wrist_idx])
            
            current_ref_landmarks.extend(hand_refs_to_check)

            if len(hand_refs_to_check) == 2: # Both wrists
                scale_dist = np.linalg.norm(hand_refs_to_check[0] - hand_refs_to_check[1])
                scale_factors[frame_idx] = scale_dist if scale_dist > 1e-6 else 1.0
            elif len(hand_refs_to_check) == 1: # One wrist
                 # Simple heuristic: use average distance of other hand points from this wrist
                 # This is a placeholder; a more robust method would be better.
                 active_hand_start_idx = left_hand_wrist_idx if not np.all(frame_landmarks[left_hand_wrist_idx]==0) else right_hand_wrist_idx
                 if active_hand_start_idx != -1:
                    other_pts_in_hand = frame_landmarks[active_hand_start_idx+1 : active_hand_start_idx+21]
                    valid_other_pts = [pt for pt in other_pts_in_hand if not np.all(pt==0)]
                    if valid_other_pts:
                        dists = np.linalg.norm(np.array(valid_other_pts) - hand_refs_to_check[0], axis=1)
                        scale_factors[frame_idx] = np.mean(dists) if np.mean(dists) > 1e-6 else 1.0
                    else: scale_factors[frame_idx] = 1.0
                 else: scale_factors[frame_idx] = 1.0
            else: # No reliable hand landmarks for scale
                scale_factors[frame_idx] = 1.0

        if current_ref_landmarks:
            reference_points[frame_idx] = np.mean(current_ref_landmarks, axis=0)
        else: # Fallback if no good reference points found
            reference_points[frame_idx] = np.mean(frame_landmarks, axis=0) # Use mean of all available landmarks
            scale_factors[frame_idx] = 1.0 # Default scale

    # STEP 2: Apply temporal smoothing
    window_size = min(5, num_frames // 2 if num_frames > 1 else 1) 
    if window_size >= 2 and num_frames >=2: # Check if smoothing is feasible
        for dim in range(3):
            reference_points[:, dim] = temporal_smooth(reference_points[:, dim], window_size)
        scale_factors = temporal_smooth(scale_factors, window_size)
    
    # STEP 3: Apply normalization
    for frame_idx in range(num_frames):
        ref_point = reference_points[frame_idx]
        scale = scale_factors[frame_idx]
        
        centered = normalized[frame_idx] - ref_point
        
        if scale > 1e-6:
            normalized[frame_idx] = centered / scale
        else:
            normalized[frame_idx] = centered # Avoid division by zero/small number
            
    if len(original_shape) == 2:
        normalized = normalized.reshape(original_shape)
    
    return normalized

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
    """Process landmarks with comprehensive normalization for environment-independent sign recognition"""
    # Replace any NaN or Inf values
    landmarks = np.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Downsample to save memory and computation
    if landmarks.shape[0] > 0: # Ensure there are frames to downsample
        landmarks = landmarks[::downsample_factor]
    
    # Apply spatial normalization to make landmarks invariant to position, scale and orientation
    # Ensure landmarks is not empty before spatial normalization
    if landmarks.shape[0] > 0:
        spatially_normalized = normalize_landmarks_spatially(landmarks)
    else:
        # If after downsampling, landmarks are empty, create zero array of expected feature dim
        # This assumes landmarks, if not empty, would have a second dimension (features)
        num_features = landmarks.shape[1] if len(landmarks.shape) > 1 and landmarks.shape[1] > 0 else 0
        # If num_features is still 0, it means the input was truly empty or 1D.
        # normalize_sequence_length expects at least shape (0, N) where N > 0 or (0,).
        # If landmarks.shape[1] was 0, then spatially_normalized is (0,0)
        # and normalize_sequence_length will try to access landmarks.shape[1] leading to an error.
        # So, if num_features is 0, we should ensure spatially_normalized has a defined feature dimension,
        # even if it's 0 frames.
        # However, normalize_landmarks_spatially itself should handle empty inputs gracefully.
        # Let's assume normalize_landmarks_spatially returns shape (0, F) or (0,) if input is (0,F) or (0,).
        spatially_normalized = landmarks # If landmarks is empty, normalize_landmarks_spatially should return it as is or an equivalent empty.

    # Use interpolation to normalize sequence length
    # normalize_sequence_length handles empty spatially_normalized if it has shape (0, features)
    # by returning np.zeros((target_length, landmarks.shape[1])).
    # This requires spatially_normalized.shape[1] to be valid.
    if spatially_normalized.shape[0] == 0 and (len(spatially_normalized.shape) < 2 or spatially_normalized.shape[1] == 0):
        # If spatially_normalized is truly empty (e.g. shape (0,) or (0,0) )
        # we cannot determine the feature dimension for normalize_sequence_length.
        # In this case, we might need to return zeros based on a known/expected feature count,
        # or propagate an error/empty array that the model can handle.
        # For now, let's assume the landmark extractor always gives at least (N, F) where F > 0, or (0, F).
        # If landmarks came in as (0,0), then spatially_normalized is (0,0).
        # The issue arises if landmarks.shape[1] is accessed on a 1D array.
        # The provided normalize_sequence_length in train.py is robust to empty landmarks.shape[0].
        # So this should be fine.
        pass # normalize_sequence_length should handle it based on its implementation.

    normalized_landmarks = normalize_sequence_length(spatially_normalized, target_length)
    
    return normalized_landmarks

def process_single_sign(video_path, target_sign, models, landmark_extractor, threshold=0.5):
    """Process a video file for a specific sign recognition (true/false)"""
    
    # Check if we have the specific binary model for the target sign
    if not isinstance(models, dict) or target_sign not in models:
        raise ValueError(f"No binary model found for sign: {target_sign}")
    
    print(f"Processing video for single sign recognition: {target_sign}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open the video file: {video_path}")
    
    # Get video properties and process frames (same as process_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0 or frame_count > 100000:
        frame_count = 300
    
    landmark_extractor.set_frame_size(width, height)
    
    # Collect landmarks from each frame
    landmarks_sequence = []
    processed_frames = 0
    max_frames = min(300, frame_count)
    
    print(f"Processing up to {max_frames} frames for {target_sign}")
    while processed_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            landmarks, results = landmark_extractor.extract_landmarks(frame)
            if landmarks is not None and np.any(landmarks):
                landmarks_sequence.append(landmarks)
            processed_frames += 1
            
            if processed_frames % 30 == 0:
                print(f"Processed {processed_frames}/{max_frames} frames")
                
        except Exception as e:
            print(f"Error extracting landmarks from frame {processed_frames}: {e}")
            continue
    
    cap.release()
    
    if not landmarks_sequence:
        raise ValueError("No landmarks could be extracted from the video. Please ensure your hands are clearly visible.")
    
    # Process landmarks
    landmarks_array = np.array(landmarks_sequence)
    print(f"Raw landmarks shape: {landmarks_array.shape}")
    
    if np.isnan(landmarks_array).any() or np.isinf(landmarks_array).any():
        print("WARNING: NaN or Inf values found in landmarks, replacing with zeros")
        landmarks_array = np.nan_to_num(landmarks_array, nan=0.0, posinf=0.0, neginf=0.0)
    
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
    
    # Run inference on the specific binary model
    model = models[target_sign]
    
    try:
        with torch.no_grad():
            input_tensor = torch.tensor(processed_landmarks, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get binary model output
            output = model(input_tensor)
            
            # Apply sigmoid to get probability (0-1 range)
            confidence = torch.sigmoid(output).item()
            
            # Determine if the sign is recognized based on threshold
            is_recognized = confidence >= threshold
            
            result = {
                "sign": target_sign,
                "recognized": is_recognized,
                "confidence": float(confidence),
                "threshold": threshold
            }
            
            print(f"Single sign recognition for '{target_sign}': {is_recognized} (confidence: {confidence:.4f}, threshold: {threshold})")
            
            return result
            
    except Exception as e:
        print(f"Error during single sign recognition for {target_sign}: {e}")
        traceback.print_exc()
        raise

def initialize_binary_models(models_dir=None):
    """Load all binary classifier models from the models directory"""
    if models_dir is None:
        # Try to import config, fall back to default if not available
        try:
            import config
            models_dir = getattr(config, 'MODELS_DIR', "./models")
        except ImportError:
            models_dir = "./models"
    
    binary_models = {}
    
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return binary_models
    
    # Get all .pth files in the models directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    print(f"Found {len(model_files)} model files: {model_files}")
    
    for model_file in model_files:
        sign_name = model_file.replace('.pth', '')
        model_path = os.path.join(models_dir, model_file)
        
        try:
            print(f"Loading binary model for '{sign_name}' from {model_path}")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Get architecture parameters with defaults for binary models
            input_size = checkpoint.get('input_size', 225)
            hidden_size = checkpoint.get('hidden_size', 128)
            num_layers = checkpoint.get('num_layers', 2)
            num_heads = checkpoint.get('num_heads', 4)
            dropout = checkpoint.get('dropout', 0.3)
            
            # Create binary classifier model
            model = BinaryTransformerClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            binary_models[sign_name] = model
            print(f"Successfully loaded binary model for '{sign_name}'")
            
        except Exception as e:
            print(f"Error loading model for {sign_name}: {e}")
            # Continue loading other models even if one fails
            continue
    
    print(f"Successfully loaded {len(binary_models)} binary models")
    return binary_models

def initialize_model(model, model_path, landmark_extractor):
    """Initialize binary models and landmark extractor"""
    global binary_models
    
    if model is None:
        print("Loading binary classifier models...")
        try:
            # Load all binary models from the models directory
            binary_models = initialize_binary_models("./models")
            
            if not binary_models:
                raise RuntimeError("No binary models could be loaded from ./models directory")
            
            print(f"Loaded {len(binary_models)} binary models for signs: {list(binary_models.keys())}")
            
            # Test all models with a dummy input
            try:
                test_input = torch.zeros((1, 150, 225), device=device)  # Default input size
                
                for sign_name, model in binary_models.items():
                    with torch.no_grad():
                        output = model(test_input)
                        prob = torch.sigmoid(output).item()
                        print(f"Test prediction for {sign_name}: {prob:.4f}")
                        
            except Exception as e:
                print(f"Error during test predictions: {e}")
                # Don't fail initialization for test errors
                
        except Exception as e:
            print(f"Error loading binary models: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Error loading binary models: {e}")
    
    if landmark_extractor is None:
        try:
            landmark_extractor = MediaPipeLandmarkExtractor()
            print("MediaPipe landmark extractor initialized")
        except Exception as e:
            print(f"Error initializing landmark extractor: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Error initializing landmark extractor: {e}")
        
    return binary_models, landmark_extractor