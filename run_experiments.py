import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np
import math
import copy

# --- I-BERT Quantization Imports ---
# Add the fairseq path to the python path to allow imports
import sys
sys.path.insert(0, 'c:\\Projects\\I-BERT')
from transformers.activations import GELUActivation
from transformers.models.vit.modeling_vit import ViTSelfAttention
from fairseq.quantization.utils.quant_modules import (
    IntGELU, IntLayerNorm, QuantAct, IntSoftmax, QuantLinear as QuantLinearBase
)

# --- Configuration ---
MODEL_NAME = "google/vit-base-patch16-224"
IMAGE_PATH = "sample_image.jpg"

# --- Wrapper Classes for Isolated Quantization ---
class QuantGELU(torch.nn.Module):
    """
    A wrapper for IntGELU that includes activation quantization (QuantAct).
    This version manually calculates the activation range to ensure quantization
    is applied correctly in a single-pass inference test.
    """
    def __init__(self):
        super().__init__()
        # Initialize QuantAct but we will override the range calculation.
        self.quant_act = QuantAct(activation_bit=8, quant_mode='symmetric', running_stat=False)
        self.int_gelu = IntGELU(quant_mode='symmetric')

    def forward(self, x):
        # Manually calculate the quantization range (min/max) from the input tensor.
        x_min = x.data.min()
        x_max = x.data.max()
        
        # Pass the dynamic range to QuantAct to get the quantized output and scaling factor.
        quant_x, scaling_factor = self.quant_act(x, specified_min=x_min, specified_max=x_max)
        
        # Apply the integer-based GELU approximation.
        output = self.int_gelu(quant_x, scaling_factor)
        return output

class QuantSoftmax(torch.nn.Module):
    """A wrapper for IntSoftmax that includes activation quantization."""
    def __init__(self, output_bit=8):
        super().__init__()
        self.quant_act = QuantAct(activation_bit=16, quant_mode='symmetric', running_stat=False)
        self.int_softmax = IntSoftmax(output_bit=output_bit, quant_mode='symmetric')

    def forward(self, x):
        x_min, x_max = x.data.min(), x.data.max()
        quant_x, scaling_factor = self.quant_act(x, specified_min=x_min, specified_max=x_max)
        return self.int_softmax(quant_x, scaling_factor)


class CustomViTSelfAttention(ViTSelfAttention):
    """Custom ViT Self-Attention layer with quantized Softmax."""
    def __init__(self, config):
        super().__init__(config)
        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)
        self.quant_softmax = QuantSoftmax(output_bit=8)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # --- Quantized Softmax --- 
        attention_probs = self.quant_softmax(attention_scores)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class QuantLinearWrapper(torch.nn.Module):
    """
    A wrapper for a linear layer that applies quantization to its activations
    and uses a quantized linear layer for the operation.
    """
    def __init__(self, fp_linear_layer):
        super().__init__()
        # Activation quantization module
        self.quant_act = QuantAct(activation_bit=8, quant_mode='symmetric', running_stat=False)
        
        # Quantized linear layer module from the library
        self.quant_linear = QuantLinearBase(weight_bit=8, bias_bit=32, quant_mode='symmetric')
        self.quant_linear.set_param(fp_linear_layer) # Copy weight/bias

    def forward(self, x):
        # Quantize activations first, calculating the dynamic range
        x_min, x_max = x.data.min(), x.data.max()
        quant_x, act_scaling_factor = self.quant_act(x, specified_min=x_min, specified_max=x_max)
        
        # Perform the quantized linear operation.
        # The library's QuantLinear returns a tuple (output_float, output_scaling_factor).
        # We only need the float tensor output for this model architecture.
        output, _ = self.quant_linear(quant_x, prev_act_scaling_factor=act_scaling_factor)
        return output


class LayerNormWrapper(torch.nn.Module):
    """Wraps IntLayerNorm to handle the scaling_factor and parameters for isolated testing."""
    def __init__(self, fp_layer):
        super().__init__()
        # The IntLayerNorm implementation requires output_bit. We'll use 8 as a standard.
        self.int_layernorm = IntLayerNorm(output_bit=8, quant_mode='symmetric')
        # Manually attach the parameters and buffer from the original FP32 layer
        self.int_layernorm.weight = fp_layer.weight
        self.int_layernorm.bias = fp_layer.bias
        self.int_layernorm.eps = fp_layer.eps

    def forward(self, x):
        # Provide a dummy scaling factor of 1.0 for isolated testing.
        return self.int_layernorm(x, scaling_factor=torch.tensor(1.0, device=x.device, dtype=x.dtype))


# --- Helper Functions ---
def get_sample_image():
    """Loads the sample image from the given path."""
    try:
        return Image.open(IMAGE_PATH)
    except FileNotFoundError:
        print(f"Error: The image file was not found at {IMAGE_PATH}")
        print("Please ensure the sample_image.jpg is in the root directory.")
        sys.exit(1)

def get_model_and_processor():
    """Loads the pre-trained ViT model and its processor."""
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor



def get_top5_predictions(logits, model):
    """Returns the top 5 predictions and their scores."""
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    top5 = torch.topk(probabilities, 5)
    
    results = []
    for i in range(5):
        label = model.config.id2label[top5.indices[0, i].item()]
        score = top5.values[0, i].item()
        results.append(f"{label} (Score: {score:.4f})")
    return results

def calculate_mse(tensor1, tensor2):
    """Calculates the Mean Squared Error between two tensors."""
    return np.mean((tensor1.detach().numpy() - tensor2.detach().numpy()) ** 2)

# --- Main Execution ---
def main():
    print("Starting ViT quantization experiments...")
    
    # 1. Load Model and Image
    model, processor = get_model_and_processor()

    image = get_sample_image()
    inputs = processor(images=image, return_tensors="pt")

    # 2. Get FP32 Baseline Results
    with torch.no_grad():
        fp32_logits = model(**inputs).logits
    
    fp32_top5 = get_top5_predictions(fp32_logits, model)
    
    print("\n--- FP32 Baseline Model ---")
    print("Top 5 Predictions:")
    for pred in fp32_top5:
        print(f"- {pred}")

    # --- Experiment 1: Quantize GELU only (Corrected with QuantAct) ---
    print("\n--- Experiment 1: FP32 + Quantized GELU (Corrected with QuantAct) ---")
    model_gelu_quantized = copy.deepcopy(model)

    # Replace GELUActivation with QuantGELU
    for name, module in model_gelu_quantized.named_modules():
        if isinstance(module, GELUActivation):
            parent_name, layer_name = name.rsplit('.', 1)
            parent_module = model_gelu_quantized.get_submodule(parent_name)
            setattr(parent_module, layer_name, QuantGELU())

    # Run inference with the GELU-quantized model
    with torch.no_grad():
        gelu_quantized_logits = model_gelu_quantized(**inputs).logits

    # Calculate and print results
    mse_gelu = calculate_mse(fp32_logits, gelu_quantized_logits)
    gelu_top5 = get_top5_predictions(gelu_quantized_logits, model)

    print(f"MSE vs. FP32: {mse_gelu:.6f}")
    print("Top 5 Predictions:")
    for pred in gelu_top5:
        print(f"- {pred}")

    # --- Experiment 2: Quantize LayerNorm only ---
    print("\n--- Experiment 2: FP32 + Quantized LayerNorm ---")
    model_ln_quantized = copy.deepcopy(model)

    # Replace LayerNorm with LayerNormWrapper
    for name, module in model_ln_quantized.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            parent_name, layer_name = name.rsplit('.', 1)
            parent_module = model_ln_quantized.get_submodule(parent_name)
            setattr(parent_module, layer_name, LayerNormWrapper(module))

    # Run inference with the LayerNorm-quantized model
    with torch.no_grad():
        ln_quantized_logits = model_ln_quantized(**inputs).logits

    # Calculate and print results
    mse_ln = calculate_mse(fp32_logits, ln_quantized_logits)
    ln_top5 = get_top5_predictions(ln_quantized_logits, model)

    print(f"MSE vs. FP32: {mse_ln:.6f}")
    print("Top 5 Predictions:")
    for pred in ln_top5:
        print(f"- {pred}")

    # --- Experiment 3: FP32 + Quantized Softmax ---
    print("\n--- Experiment 3: FP32 + Quantized Softmax ---")
    model_softmax_quantized = copy.deepcopy(model)
    for name, module in model_softmax_quantized.named_modules():
        if isinstance(module, ViTSelfAttention):
            # The config is needed to initialize the custom attention module
            custom_attention = CustomViTSelfAttention(model_softmax_quantized.config)
            
            # To replace the module, we need to get its parent
            parent_name, layer_name = name.rsplit('.', 1)
            parent_module = model_softmax_quantized.get_submodule(parent_name)
            
            # Set the new custom module
            setattr(parent_module, layer_name, custom_attention)

    # Run inference with the Softmax-quantized model
    with torch.no_grad():
        softmax_quantized_logits = model_softmax_quantized(**inputs).logits

    # Calculate and print results
    mse_softmax = calculate_mse(fp32_logits, softmax_quantized_logits)
    softmax_top5 = get_top5_predictions(softmax_quantized_logits, model)

    print(f"MSE vs. FP32: {mse_softmax:.6f}")
    print("Top 5 Predictions:")
    for pred in softmax_top5:
        print(f"- {pred}")

    # --- Experiment 4: FP32 + Quantized Linear Layers ---
    print("\n--- Experiment 4: FP32 + Quantized Linear Layers ---")
    model_linear_quantized = copy.deepcopy(model)
    for name, module in model_linear_quantized.named_modules():
        if isinstance(module, torch.nn.Linear):
            custom_linear = QuantLinearWrapper(module)
            if '.' in name:
                parent_name, layer_name = name.rsplit('.', 1)
                parent_module = model_linear_quantized.get_submodule(parent_name)
            else:
                # This handles top-level modules like the final classifier layer
                parent_module = model_linear_quantized
                layer_name = name
            setattr(parent_module, layer_name, custom_linear)

    # Run inference with the linear-quantized model
    with torch.no_grad():
        linear_quantized_logits = model_linear_quantized(**inputs).logits

    # Calculate and print results
    mse_linear = calculate_mse(fp32_logits, linear_quantized_logits)
    linear_top5 = get_top5_predictions(linear_quantized_logits, model)

    print(f"MSE vs. FP32: {mse_linear:.6f}")
    print("Top 5 Predictions:")
    for pred in linear_top5:
        print(f"- {pred}")

    # --- Experiment 5: Quantized Linear + GELU ---
    print("\n--- Experiment 5: Quantized Linear + GELU ---")
    model_linear_gelu_quantized = copy.deepcopy(model)
    for name, module in model_linear_gelu_quantized.named_modules():
        # Replace Linear layers
        if isinstance(module, torch.nn.Linear):
            custom_linear = QuantLinearWrapper(module)
            if '.' in name:
                parent_name, layer_name = name.rsplit('.', 1)
                parent_module = model_linear_gelu_quantized.get_submodule(parent_name)
            else:
                parent_module = model_linear_gelu_quantized
                layer_name = name
            setattr(parent_module, layer_name, custom_linear)
        
        # Replace GELU layers
        if isinstance(module, GELUActivation):
            quant_gelu = QuantGELU()
            parent_name, layer_name = name.rsplit('.', 1)
            parent_module = model_linear_gelu_quantized.get_submodule(parent_name)
            setattr(parent_module, layer_name, quant_gelu)

    # Run inference
    with torch.no_grad():
        linear_gelu_logits = model_linear_gelu_quantized(**inputs).logits

    # Calculate and print results
    mse_linear_gelu = calculate_mse(fp32_logits, linear_gelu_logits)
    linear_gelu_top5 = get_top5_predictions(linear_gelu_logits, model)

    print(f"MSE vs. FP32: {mse_linear_gelu:.6f}")
    print("Top 5 Predictions:")
    for pred in linear_gelu_top5:
        print(f"- {pred}")

if __name__ == "__main__":
    main()
