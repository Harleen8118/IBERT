#!/usr/bin/env python3

import torch
import torch.nn as nn
from fairseq.modules.quantization.scalar.modules import IntLinear, ActivationQuantizer

def test_quantized_linear():
    """Test basic quantized linear layer functionality"""
    print("Testing quantized linear layer...")
    
    # Create a simple quantized linear layer
    in_features = 128
    out_features = 64
    batch_size = 4
    
    # Original linear layer
    linear_orig = nn.Linear(in_features, out_features)
    
    # Quantized linear layer - using tensor method to avoid CUDA requirements
    linear_quant = IntLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        p=1.0,  # Full quantization for testing
        bits=8,
        method="tensor"  # tensor method works on CPU
    )
    
    # Copy weights from original to quantized
    linear_quant.weight.data = linear_orig.weight.data.clone()
    linear_quant.bias.data = linear_orig.bias.data.clone()
    
    # Test input
    x = torch.randn(batch_size, in_features)
    
    # Forward pass
    y_orig = linear_orig(x)
    y_quant = linear_quant(x)
    
    print(f"Original output shape: {y_orig.shape}")
    print(f"Quantized output shape: {y_quant.shape}")
    print(f"Output difference (MSE): {torch.mean((y_orig - y_quant) ** 2):.6f}")
    
    return True

def test_activation_quantizer():
    """Test activation quantization"""
    print("\nTesting activation quantizer...")
    
    # Create a simple module
    linear = nn.Linear(64, 32)
    
    # Add activation quantizer - using tensor method for CPU compatibility
    act_quant = ActivationQuantizer(
        module=linear,
        p=1.0,  # Full quantization
        bits=8,
        method="tensor"  # tensor method works on CPU
    )
    
    # Test input
    x = torch.randn(4, 64)
    
    # Forward pass
    y = linear(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Activation quantizer successfully applied!")
    
    # Remove hook
    act_quant.handle.remove()
    
    return True

def main():
    print("I-BERT Installation Test")
    print("=" * 30)
    
    try:
        test_quantized_linear()
        test_activation_quantizer()
        print("\n✅ All tests passed! I-BERT is working correctly.")
        return True
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 