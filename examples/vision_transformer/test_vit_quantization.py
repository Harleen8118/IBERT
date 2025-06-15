#!/usr/bin/env python3

import torch
import torch.nn as nn
from transformers import ViTModel, ViTForImageClassification, ViTImageProcessor
from PIL import Image
import requests
import io

def test_vit_model_loading():
    """Test loading the ViT model and basic inference"""
    print("Testing ViT model loading and inference...")
    
    # Load the model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    # Load a test image
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    try:
        image = Image.open(requests.get(url, stream=True).raw)
    except:
        # Create a dummy image if internet is not available
        print("Creating dummy image for testing...")
        import numpy as np
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(dummy_image)
    
    # Process the image and run inference
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    
    print(f"Model loaded successfully!")
    print(f"Input shape: {inputs['pixel_values'].shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Predicted class index: {predicted_class_idx}")
    
    return model, processor

def explore_vit_architecture(model):
    """Explore the ViT model architecture to understand the layers"""
    print("\nExploring ViT model architecture...")
    
    print(f"Model type: {type(model).__name__}")
    print(f"Config: {model.config}")
    
    print("\nModel structure:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            print(f"  {name}: {type(module).__name__} - {module}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    return

def test_basic_quantization_approach(model):
    """Test applying basic quantization to linear layers in ViT"""
    print("\nTesting basic quantization approach...")
    
    from fairseq.modules.quantization.scalar.modules import IntLinear
    
    # Find all linear layers
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))
    
    print(f"Found {len(linear_layers)} linear layers:")
    for name, layer in linear_layers[:5]:  # Show first 5
        print(f"  {name}: {layer.in_features} -> {layer.out_features}")
    
    # Try to replace one linear layer with quantized version
    if linear_layers:
        name, original_layer = linear_layers[0]
        print(f"\nTesting quantization on layer: {name}")
        
        # Create quantized version
        quant_layer = IntLinear(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            bias=original_layer.bias is not None,
            p=1.0,  # Full quantization
            bits=8,
            method="tensor"
        )
        
        # Copy weights
        quant_layer.weight.data = original_layer.weight.data.clone()
        if original_layer.bias is not None:
            quant_layer.bias.data = original_layer.bias.data.clone()
        
        print(f"Original layer: {original_layer}")
        print(f"Quantized layer: {quant_layer}")
        
        # Test with dummy input
        dummy_input = torch.randn(1, original_layer.in_features)
        
        with torch.no_grad():
            original_output = original_layer(dummy_input)
            quant_output = quant_layer(dummy_input)
            
        diff = torch.mean((original_output - quant_output) ** 2)
        print(f"MSE difference: {diff:.6f}")
        
        return True
    
    return False

def test_full_model_quantization(model, processor):
    """Quantize the entire ViT model (8-bit tensor) and compare logits"""
    print("\nTesting full-model quantization (8-bit tensor)...")

    from fairseq.models.vit_quantization import ViTQuantizer

    # Build quantizer (CPU-friendly settings)
    quantizer = ViTQuantizer(
        bits=8,
        method="tensor",   # Works on CPU
        p=1.0,
        quantize_attention=True,
        quantize_mlp=True,
        quantize_classifier=True,
    )

    # Quantize the model (this may take ~1-2 min on CPU)
    q_model = quantizer.quantize_model(model)

    # Create a dummy or real image as in the loading test
    try:
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        image = Image.open(requests.get(url, stream=True).raw)
    except Exception:
        from PIL import Image
        import numpy as np
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(dummy)

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        logits_fp = model(**inputs).logits
        logits_int = q_model(**inputs).logits

    mse = torch.mean((logits_fp - logits_int) ** 2).item()
    top1_fp = logits_fp.argmax(-1).item()
    top1_int = logits_int.argmax(-1).item()

    print(f"Original logits shape: {logits_fp.shape}")
    print(f"Quantized logits shape: {logits_int.shape}")
    print(f"Logits MSE difference: {mse:.6f}")
    print(f"Top-1 match: {top1_fp == top1_int}")

    # Return success flag
    return top1_fp == top1_int

def main():
    print("ViT Quantization Test")
    print("=" * 40)
    
    try:
        # Test 1: Load ViT model
        model, processor = test_vit_model_loading()
        
        # # Test 2: Explore architecture
        # explore_vit_architecture(model)
        
        # Test 3: Test basic quantization
        success = test_basic_quantization_approach(model)
        
        # Test 4: Full model quantization shape & diff test
        full_success = test_full_model_quantization(model, processor)
        
        if success and full_success:
            print("\n✅ All ViT quantization tests successful!")
        else:
            print("\n❌ Quantization test failed")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
