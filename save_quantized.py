import torch
from transformers import AutoModelForImageClassification
from torch.quantization import quantize_dynamic
import argparse
import os

def save_quantized_model(model_name: str, output_path: str):
    """
    Loads a pre-trained model, applies dynamic quantization, and saves the quantized model.

    Args:
        model_name (str): The name of the pre-trained model from Hugging Face.
        output_path (str): The path to save the quantized model file.
    """
    print(f"Loading pre-trained model: {model_name}")
    model = AutoModelForImageClassification.from_pretrained(model_name)

    print("Applying dynamic quantization...")
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},  # Layers to quantize
        dtype=torch.qint8   # Target data type for weights
    )

    print(f"Saving quantized model to: {output_path}")
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(quantized_model.state_dict(), output_path)
    print("Quantized model saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Quantize and save a pre-trained model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vit-base-patch16-224",
        help="The name of the pre-trained model from Hugging Face."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/quantized/vit_base_quantized.pt",
        help="The path to save the quantized model file."
    )
    args = parser.parse_args()

    save_quantized_model(args.model_name, args.output_path)

if __name__ == "__main__":
    main()
