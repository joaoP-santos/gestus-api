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
            target_length=136,
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
                    raise FileNotFoundError(f"Model file not found at {model_path}. Make sure the model file is included or set MODEL_PATH appropriately.")
            
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
                raise RuntimeError(f"Error loading state_dict: {e}")
                
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
                raise RuntimeError(f"Error during test prediction: {e}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Error loading model: {e}")
            
    if landmark_extractor is None:
        try:
            landmark_extractor = MediaPipeLandmarkExtractor()
            print("MediaPipe landmark extractor initialized")
        except Exception as e:
            print(f"Error initializing landmark extractor: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Error initializing landmark extractor: {e}")
        
    return model, landmark_extractor
