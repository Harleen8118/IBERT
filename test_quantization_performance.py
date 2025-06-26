import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import sys

# --- I-BERT Imports ---
sys.path.insert(0, 'c:\\Users\\klinu\\Desktop\\I-BERT')
from run_ibert_quantization import ViTQuantizer, MODEL_NAME, NUM_LABELS, FP32_FINETUNED_PATH, CALIBRATED_MODEL_PATH, QAT_MODEL_PATH

# --- Configuration ---
IMAGE_PATH = "sample_image.jpg"

# --- Helper Functions ---
def get_sample_image():
    try:
        return Image.open(IMAGE_PATH).convert("RGB")
    except FileNotFoundError:
        print(f"Error: The image file was not found at {IMAGE_PATH}")
        print("Please ensure sample_image.jpg is in the root directory.")
        sys.exit(1)

def get_top5_predictions(logits, model_config):
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    top5 = torch.topk(probabilities, 5)
    results = []
    for i in range(5):
        label = model_config.id2label[top5.indices[0, i].item()]
        score = top5.values[0, i].item()
        results.append(f"{label} (Score: {score:.4f})")
    return results

def calculate_mse(tensor1, tensor2):
    return np.mean((tensor1.detach().cpu().numpy() - tensor2.detach().cpu().numpy()) ** 2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Processor and Image
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    image = get_sample_image()
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 2. Load All Model Versions
    print("--- Loading all model versions ---")

    # Original FP32 Model
    model_fp32_orig = ViTForImageClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True).to(device)
    model_fp32_orig.eval()

    # Fine-tuned FP32 Model
    model_fp32_finetuned = ViTForImageClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
    model_fp32_finetuned.load_state_dict(torch.load(FP32_FINETUNED_PATH, map_location=device))
    model_fp32_finetuned.to(device)
    model_fp32_finetuned.eval()

    # Base model for creating quantized versions
    base_model = ViTForImageClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
    quantizer = ViTQuantizer(quant_mode='symmetric', bits=8)

    # Calibrated Quantized Model
    model_calibrated = quantizer.quantize(base_model)
    model_calibrated.load_state_dict(torch.load(CALIBRATED_MODEL_PATH, map_location=device))
    model_calibrated.to(device)
    model_calibrated.eval()

    # QAT Quantized Model
    model_qat = quantizer.quantize(base_model) # Re-quantize a base model to get the structure
    model_qat.load_state_dict(torch.load(QAT_MODEL_PATH, map_location=device))
    model_qat.to(device)
    model_qat.eval()

    print("--- All models loaded successfully ---")

    # 3. Get Predictions and Compare
    with torch.no_grad():
        logits_fp32_orig = model_fp32_orig(**inputs).logits
        logits_fp32_finetuned = model_fp32_finetuned(**inputs).logits
        logits_calibrated = model_calibrated(**inputs).logits
        logits_qat = model_qat(**inputs).logits

    # --- Print Results ---
    print("\n--- Comparison of Model Predictions ---")

    # Original Model
    preds_fp32_orig = get_top5_predictions(logits_fp32_orig, model_fp32_orig.config)
    print("\n1. Original FP32 Model Predictions:")
    for pred in preds_fp32_orig:
        print(f"- {pred}")

    # Fine-tuned Model
    preds_fp32_finetuned = get_top5_predictions(logits_fp32_finetuned, model_fp32_finetuned.config)
    print("\n2. Fine-tuned FP32 Model Predictions:")
    for pred in preds_fp32_finetuned:
        print(f"- {pred}")

    # Calibrated Model
    preds_calibrated = get_top5_predictions(logits_calibrated, model_calibrated.config)
    mse_calibrated = calculate_mse(logits_fp32_finetuned, logits_calibrated)
    print("\n3. Calibrated Quantized Model Predictions:")
    print(f"   MSE vs. Fine-tuned FP32: {mse_calibrated:.6f}")
    for pred in preds_calibrated:
        print(f"- {pred}")

    # QAT Model
    preds_qat = get_top5_predictions(logits_qat, model_qat.config)
    mse_qat = calculate_mse(logits_fp32_finetuned, logits_qat)
    print("\n4. QAT Quantized Model Predictions:")
    print(f"   MSE vs. Fine-tuned FP32: {mse_qat:.6f}")
    for pred in preds_qat:
        print(f"- {pred}")

if __name__ == "__main__":
    main()
