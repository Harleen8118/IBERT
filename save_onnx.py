import argparse
import os
from pathlib import Path
from optimum.onnxruntime import ORTQuantizer, ORTModelForImageClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig

def main():
    parser = argparse.ArgumentParser(description="Quantize a Hugging Face model and save it in ONNX format.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vit-base-patch16-224",
        help="The name of the pre-trained model from Hugging Face."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="onnx_models/quantized",
        help="The directory to save the quantized ONNX model."
    )
    args = parser.parse_args()

    # Define the output directory and ensure it exists
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the pre-trained model
    print(f"Loading and exporting model: {args.model_name}")
    model = ORTModelForImageClassification.from_pretrained(args.model_name, export=True)
    
    # Create a quantizer from the model
    quantizer = ORTQuantizer.from_pretrained(model)
    
    # Define the quantization configuration
    print("Applying dynamic quantization...")
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    
    # Apply quantization
    quantizer.quantize(
        save_dir=output_dir,
        quantization_config=qconfig,
    )
    
    print(f"\nQuantized model saved to: {output_dir}")
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main()
