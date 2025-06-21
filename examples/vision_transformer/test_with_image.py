#!/usr/bin/env python3

"""
Test ViT Quantization with a specific image
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os
import sys
import numpy as np

# Add the current directory to the path to import our modules
sys.path.append('.')

from fairseq.models.vit_quantization import ViTQuantizer, compare_models

def load_and_process_image(image_path, processor):
    """Load and process the specified image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    inputs = processor(images=image, return_tensors="pt")
    print(f"Processed input shape: {inputs['pixel_values'].shape}")
    
    return image, inputs

def get_class_name(class_idx, model):
    """Get the class name for a prediction index"""
    if hasattr(model.config, 'id2label'):
        return model.config.id2label.get(class_idx, f"Class {class_idx}")
    return f"Class {class_idx}"

def test_with_image(image_path):
    """Test ViT quantization with the specified image"""
    print("ViT Quantization Test with Custom Image")
    print("=" * 50)
    
    # Load model and processor
    print("Loading ViT model and processor...")
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load and process the image
    try:
        image, inputs = load_and_process_image(image_path, processor)
    except Exception as e:
        print(f"Error loading image: {e}")
        return False
    
    # Test original model
    print("\nTesting original model...")
    model.eval()
    with torch.no_grad():
        original_outputs = model(**inputs)
        original_logits = original_outputs.logits
        original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
        original_pred = original_logits.argmax(-1).item()
        original_confidence = original_probs.max().item()
    
    original_class = get_class_name(original_pred, model)
    print(f"Original prediction: {original_class} (confidence: {original_confidence:.3f})")
    print(f"Original logits (top 5): {original_logits.squeeze()[:5].numpy()}")
    
    # Show top 5 predictions
    top5_indices = original_probs.squeeze().argsort(descending=True)[:5]
    print("Top 5 predictions (original):")
    for i, idx in enumerate(top5_indices):
        class_name = get_class_name(idx.item(), model)
        confidence = original_probs.squeeze()[idx].item()
        print(f"  {i+1}. {class_name}: {confidence:.3f}")
    
    # Initialize quantizer with different configurations
    print("\nTesting different quantization configurations...")
    
    configs = [
        {"bits": 8, "method": "tensor", "p": 1.0, "name": "8-bit tensor"},
        #{"bits": 8, "method": "histogram", "p": 1.0, "name": "8-bit histogram"},
        {"bits": 4, "method": "tensor", "p": 1.0, "name": "4-bit tensor"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n--- Testing {config['name']} quantization ---")
        
        # Initialize quantizer
        quantizer = ViTQuantizer(
            bits=config["bits"],
            method=config["method"],
            p=config["p"],
            quantize_attention=True,
            quantize_mlp=True,
            quantize_classifier=True,
            quantize_nonlinear=True
        )
        
        # Quantize the model
        print("Quantizing model...")
        quantized_model = quantizer.quantize_model(model)
        
        # Test quantized model
        print("Testing quantized model...")
        quantized_model.eval()
        with torch.no_grad():
            quantized_outputs = quantized_model(**inputs)
            quantized_logits = quantized_outputs.logits
            quantized_probs = torch.nn.functional.softmax(quantized_logits, dim=-1)
            quantized_pred = quantized_logits.argmax(-1).item()
            quantized_confidence = quantized_probs.max().item()
        
        quantized_class = get_class_name(quantized_pred, model)
        print(f"Quantized prediction: {quantized_class} (confidence: {quantized_confidence:.3f})")
        print(f"Quantized logits (top 5): {quantized_logits.squeeze()[:5].numpy()}")
        
        # Compare models
        comparison = compare_models(model, quantized_model, processor, inputs)
        
        if comparison:
            result = {
                'config': config['name'],
                'mse': comparison['mse'],
                'mae': comparison['mae'],
                'top1_match': comparison['top1_accuracy_match'],
                'original_pred': original_pred,
                'quantized_pred': quantized_pred,
                'original_class': original_class,
                'quantized_class': quantized_class,
                'original_confidence': original_confidence,
                'quantized_confidence': quantized_confidence
            }
            results.append(result)
            
            print(f"MSE Loss: {comparison['mse']:.6f}")
            print(f"MAE Loss: {comparison['mae']:.6f}")
            print(f"Top-1 Match: {comparison['top1_accuracy_match']:.2%}")
            print(f"Prediction Match: {'✅' if original_pred == quantized_pred else '❌'}")
            # Add layer-wise MSE if available
            if 'layer_mse' in comparison:
                print("Layer-wise MSE:")
                for layer, mse in comparison['layer_mse'].items():
                    print(f"  {layer}: {mse:.6f}")
        
        # Clean up
        quantizer.remove_activation_quantizers()
        del quantized_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Summary
    print("\n" + "=" * 50)
    print("QUANTIZATION RESULTS SUMMARY")
    print("=" * 50)
    
    for result in results:
        match_symbol = "✅" if result['original_pred'] == result['quantized_pred'] else "❌"
        print(f"{result['config']:15} | MSE: {result['mse']:.6f} | "
              f"Pred: {match_symbol} | Conf: {result['quantized_confidence']:.3f}")
    
    print(f"\nOriginal: {original_class} ({original_confidence:.3f})")
    
    return True

def main():
    image_path = "sample_image.jpg"
    
    try:
        success = test_with_image(image_path)
        if success:
            print("\n🎉 Test completed successfully!")
        else:
            print("\n❌ Test failed!")
            return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())