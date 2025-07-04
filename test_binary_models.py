#!/usr/bin/env python3
"""
Test script to verify binary model loading and inference
"""

import os
import sys
import torch
import numpy as np

# Add the current directory to Python path
sys.path.append('.')

from utils import initialize_binary_models, BinaryTransformerClassifier

def test_binary_models():
    """Test loading and running inference on binary models"""
    
    print("Testing binary model loading...")
    
    # Test 1: Check if models directory exists
    models_dir = "./models"
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return False
    
    # Test 2: List available model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    print(f"✅ Found {len(model_files)} model files: {model_files}")
    
    if not model_files:
        print("❌ No .pth model files found in models directory")
        return False
    
    # Test 3: Load binary models
    try:
        binary_models = initialize_binary_models(models_dir)
        print(f"✅ Successfully loaded {len(binary_models)} binary models")
        
        if not binary_models:
            print("❌ No binary models were loaded successfully")
            return False
            
        # Print loaded models
        for sign_name in binary_models.keys():
            print(f"  - {sign_name}")
            
    except Exception as e:
        print(f"❌ Error loading binary models: {e}")
        return False
    
    # Test 4: Test inference on each model
    print("\nTesting inference...")
    try:
        # Create dummy input (batch_size=1, sequence_length=150, features=225)
        dummy_input = torch.zeros((1, 150, 225))
        
        results = {}
        for sign_name, model in binary_models.items():
            with torch.no_grad():
                output = model(dummy_input)
                prob = torch.sigmoid(output).item()
                results[sign_name] = prob
                print(f"  {sign_name}: {prob:.4f}")
        
        print(f"✅ Successfully ran inference on all {len(results)} models")
        
        # Test 5: Sort by confidence
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 3 predictions (sorted by confidence):")
        for i, (sign, conf) in enumerate(sorted_results[:3]):
            print(f"  {i+1}. {sign}: {conf:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during inference testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_architecture():
    """Test the BinaryTransformerClassifier architecture"""
    print("\nTesting BinaryTransformerClassifier architecture...")
    
    try:
        # Create a model instance
        model = BinaryTransformerClassifier(
            input_size=225,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            dropout=0.3
        )
        
        # Test forward pass
        dummy_input = torch.randn(2, 150, 225)  # batch_size=2
        
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"✅ Model forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # Test sigmoid activation
        probs = torch.sigmoid(output)
        print(f"  Sigmoid range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model architecture: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting binary model tests...\n")
    
    # Test model architecture
    arch_test = test_model_architecture()
    
    # Test binary model loading
    models_test = test_binary_models()
    
    print("\n" + "="*50)
    print("TEST RESULTS:")
    print(f"Architecture test: {'✅ PASSED' if arch_test else '❌ FAILED'}")
    print(f"Binary models test: {'✅ PASSED' if models_test else '❌ FAILED'}")
    
    if arch_test and models_test:
        print("\n🎉 All tests passed! Binary models are ready to use.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        sys.exit(1)
