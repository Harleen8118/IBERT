import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.quantization import quantize_dynamic
import torch.onnx
import os

def convert_to_onnx(model, model_name, output_dir="onnx_models", quantized=False):
    """
    Convert a PyTorch model to ONNX format.
    
    Args:
        model: The PyTorch model to convert
        model_name: Name to save the ONNX model
        output_dir: Directory to save the ONNX model
        quantized: Whether the model is quantized (for naming purposes)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Create dummy input (batch_size, num_channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Define output file path
    suffix = "_quantized" if quantized else ""
    output_path = os.path.join(output_dir, f"{model_name}{suffix}.onnx")
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,  # ONNX opset version
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model saved to {output_path}")
    return output_path

def main():
    # Model configuration
    model_name = "google/vit-base-patch16-224"
    output_dir = "onnx_models"
    
    print(f"Loading model {model_name}...")
    
    # Load the pre-trained model and feature extractor
    model = ViTForImageClassification.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    
    # Convert original model to ONNX
    print("\nConverting original model to ONNX...")
    convert_to_onnx(model, "vit_base_patch16_224", output_dir, quantized=False)
    
    # Create a quantized version of the model
    print("\nCreating quantized model...")
    quantized_model = quantize_dynamic(
        model,  # original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8  # the target dtype for quantized weights
    )
    
    # Convert quantized model to ONNX
    print("Converting quantized model to ONNX...")
    convert_to_onnx(quantized_model, "vit_base_patch16_224", output_dir, quantized=True)
    
    print("\nConversion completed successfully!")

if __name__ == "__main__":
    main()
