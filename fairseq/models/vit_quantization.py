#!/usr/bin/env python3

"""
ViT Quantization Framework
Adapts I-BERT's integer quantization approach for Vision Transformer models.
"""

import torch
import torch.nn as nn
import copy
import logging
from typing import Dict, List, Tuple, Optional, Union
from transformers import ViTModel, ViTForImageClassification, ViTImageProcessor
from fairseq.modules.quantization.scalar.modules import IntLinear, ActivationQuantizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantizedViTAttention(nn.Module):
    """Quantized version of ViT attention mechanism"""
    
    def __init__(self, original_attention, quant_config):
        super().__init__()
        self.config = original_attention.config
        self.num_attention_heads = self.config.num_attention_heads
        self.attention_head_size = int(self.config.hidden_size / self.config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Quantized linear layers
        self.query = IntLinear(
            in_features=original_attention.query.in_features,
            out_features=original_attention.query.out_features,
            bias=original_attention.query.bias is not None,
            **quant_config
        )
        
        self.key = IntLinear(
            in_features=original_attention.key.in_features,
            out_features=original_attention.key.out_features,
            bias=original_attention.key.bias is not None,
            **quant_config
        )
        
        self.value = IntLinear(
            in_features=original_attention.value.in_features,
            out_features=original_attention.value.out_features,
            bias=original_attention.value.bias is not None,
            **quant_config
        )
        
        # Copy weights from original
        self.query.weight.data = original_attention.query.weight.data.clone()
        if original_attention.query.bias is not None:
            self.query.bias.data = original_attention.query.bias.data.clone()
            
        self.key.weight.data = original_attention.key.weight.data.clone()
        if original_attention.key.bias is not None:
            self.key.bias.data = original_attention.key.bias.data.clone()
            
        self.value.weight.data = original_attention.value.weight.data.clone()
        if original_attention.value.bias is not None:
            self.value.bias.data = original_attention.value.bias.data.clone()
        
        # Dropout
        self.dropout = original_attention.dropout
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class ViTQuantizer:
    """Main quantization class for ViT models"""
    
    def __init__(self, 
                 bits: int = 8,
                 method: str = "tensor",
                 p: float = 1.0,
                 update_step: int = 1000,
                 quantize_embeddings: bool = True,
                 quantize_attention: bool = True,
                 quantize_mlp: bool = True,
                 quantize_classifier: bool = True,
                 quantize_activations: bool = False):
        """
        Initialize ViT quantizer
        
        Args:
            bits: Number of bits for quantization
            method: Quantization method ("tensor", "histogram", "channel")
            p: Quantization noise probability (0 = no quantization, 1 = full quantization)
            update_step: Steps between quantization parameter updates
            quantize_embeddings: Whether to quantize embedding layers
            quantize_attention: Whether to quantize attention layers
            quantize_mlp: Whether to quantize MLP layers
            quantize_classifier: Whether to quantize classifier head
            quantize_activations: Whether to quantize activations
        """
        self.quant_config = {
            'bits': bits,
            'method': method,
            'p': p,
            'update_step': update_step
        }
        
        self.quantize_embeddings = quantize_embeddings
        self.quantize_attention = quantize_attention
        self.quantize_mlp = quantize_mlp
        self.quantize_classifier = quantize_classifier
        self.quantize_activations = quantize_activations
        
        self.activation_quantizers = []
        
    def quantize_linear_layer(self, layer: nn.Linear) -> IntLinear:
        """Replace a linear layer with its quantized version"""
        quantized_layer = IntLinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            **self.quant_config
        )
        
        # Copy weights
        quantized_layer.weight.data = layer.weight.data.clone()
        if layer.bias is not None:
            quantized_layer.bias.data = layer.bias.data.clone()
            
        return quantized_layer
    
    def quantize_vit_encoder_layer(self, layer):
        """Quantize a single ViT encoder layer"""
        # Quantize attention layers
        if self.quantize_attention:
            # Query, Key, Value projections
            layer.attention.attention.query = self.quantize_linear_layer(
                layer.attention.attention.query
            )
            layer.attention.attention.key = self.quantize_linear_layer(
                layer.attention.attention.key
            )
            layer.attention.attention.value = self.quantize_linear_layer(
                layer.attention.attention.value
            )
            
            # Attention output projection
            layer.attention.output.dense = self.quantize_linear_layer(
                layer.attention.output.dense
            )
            
            # Add activation quantization for attention if enabled
            if self.quantize_activations:
                self.activation_quantizers.append(
                    ActivationQuantizer(
                        module=layer.attention,
                        **self.quant_config
                    )
                )
        
        # Quantize MLP layers
        if self.quantize_mlp:
            layer.intermediate.dense = self.quantize_linear_layer(
                layer.intermediate.dense
            )
            layer.output.dense = self.quantize_linear_layer(
                layer.output.dense
            )
            
            # Add activation quantization for MLP if enabled
            if self.quantize_activations:
                self.activation_quantizers.append(
                    ActivationQuantizer(
                        module=layer.intermediate,
                        **self.quant_config
                    )
                )
    
    def quantize_model(self, model: Union[ViTModel, ViTForImageClassification]) -> Union[ViTModel, ViTForImageClassification]:
        """
        Quantize the entire ViT model
        
        Args:
            model: The ViT model to quantize
            
        Returns:
            Quantized model
        """
        logger.info(f"Starting ViT model quantization with {self.quant_config['bits']} bits")
        
        # Create a deep copy to avoid modifying the original model
        quantized_model = copy.deepcopy(model)
        
        # Quantize embedding layers
        if self.quantize_embeddings and hasattr(quantized_model.vit.embeddings, 'patch_embeddings'):
            # Note: patch_embeddings uses Conv2d, not Linear, so we handle it differently
            # For now, we skip Conv2d quantization as I-BERT doesn't have Conv2d quantization
            logger.info("Skipping Conv2d patch embedding quantization (not implemented)")
        
        # Quantize encoder layers
        if hasattr(quantized_model.vit, 'encoder'):
            for i, layer in enumerate(quantized_model.vit.encoder.layer):
                logger.info(f"Quantizing encoder layer {i}")
                self.quantize_vit_encoder_layer(layer)
        
        # Quantize classifier head
        if self.quantize_classifier and hasattr(quantized_model, 'classifier'):
            logger.info("Quantizing classifier head")
            quantized_model.classifier = self.quantize_linear_layer(
                quantized_model.classifier
            )
        
        logger.info(f"Quantization complete. Added {len(self.activation_quantizers)} activation quantizers")
        return quantized_model
    
    def remove_activation_quantizers(self):
        """Remove all activation quantizers"""
        for quantizer in self.activation_quantizers:
            if hasattr(quantizer, 'handle') and quantizer.handle is not None:
                quantizer.handle.remove()
        self.activation_quantizers.clear()

def compare_models(original_model, quantized_model, processor, test_input=None):
    """Compare outputs of original and quantized models"""
    if test_input is None:
        # Create dummy test input
        import numpy as np
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        from PIL import Image
        test_image = Image.fromarray(dummy_image)
        test_input = processor(images=test_image, return_tensors="pt")
    
    # Set models to eval mode
    original_model.eval()
    quantized_model.eval()
    
    with torch.no_grad():
        # Get outputs
        original_outputs = original_model(**test_input)
        quantized_outputs = quantized_model(**test_input)
        
        # Compare logits
        if hasattr(original_outputs, 'logits') and hasattr(quantized_outputs, 'logits'):
            original_logits = original_outputs.logits
            quantized_logits = quantized_outputs.logits
            
            # Calculate differences
            mse = torch.mean((original_logits - quantized_logits) ** 2)
            mae = torch.mean(torch.abs(original_logits - quantized_logits))
            
            # Calculate top-1 accuracy difference
            orig_pred = original_logits.argmax(-1)
            quant_pred = quantized_logits.argmax(-1)
            top1_match = (orig_pred == quant_pred).float().mean()
            
            return {
                'mse': mse.item(),
                'mae': mae.item(),
                'top1_accuracy_match': top1_match.item(),
                'original_pred': orig_pred.item(),
                'quantized_pred': quant_pred.item()
            }
    
    return None

def main():
    """Demo of ViT quantization"""
    print("ViT Quantization Framework Demo")
    print("=" * 50)
    
    # Load model and processor
    print("Loading ViT model...")
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize quantizer
    quantizer = ViTQuantizer(
        bits=8,
        method="tensor",
        p=1.0,  # Full quantization
        quantize_attention=True,
        quantize_mlp=True,
        quantize_classifier=True,
        quantize_activations=False  # Disable for now
    )
    
    # Quantize the model
    print("Quantizing model...")
    quantized_model = quantizer.quantize_model(model)
    
    print(f"Quantized model parameters: {sum(p.numel() for p in quantized_model.parameters()):,}")
    
    # Compare models
    print("Comparing original vs quantized model...")
    comparison = compare_models(model, quantized_model, processor)
    
    if comparison:
        print(f"MSE Loss: {comparison['mse']:.6f}")
        print(f"MAE Loss: {comparison['mae']:.6f}")
        print(f"Top-1 Prediction Match: {comparison['top1_accuracy_match']:.2%}")
        print(f"Original Prediction: {comparison['original_pred']}")
        print(f"Quantized Prediction: {comparison['quantized_pred']}")
        
        if comparison['top1_accuracy_match'] > 0.9:
            print("✅ Quantization successful! Predictions closely match.")
        else:
            print("⚠️  Quantization may have affected model accuracy.")
    
    # Clean up
    quantizer.remove_activation_quantizers()
    
    print("\nQuantization demo complete!")
    return quantized_model

if __name__ == "__main__":
    main() 