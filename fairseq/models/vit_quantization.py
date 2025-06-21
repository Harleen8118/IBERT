#!/usr/bin/env python3

"""ViT integer-only quantisation (I-BERT style).
This file replaces the previous draft and keeps the implementation CPU/GPU-agnostic.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTModel

# I-BERT building blocks -------------------------------------------------------
from fairseq.quantization.utils.quant_modules import (
    IntGELU,
    IntLayerNorm,
    IntSoftmax,
    QuantAct,
    QuantLinear,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Helper wrappers
# -----------------------------------------------------------------------------

class IntGELUWrapper(nn.Module):
    """Bridges ViT GELU interface with I-BERT IntGELU.
    Supports both standard ViT call (tensor only) *and* `(tensor, sf)` when
    scaling-factor propagation is enabled (Phase 2).
    """

    def __init__(self, quant_mode: str = "none", bits: int = 8):
        super().__init__()
        self.int_gelu = IntGELU(quant_mode=quant_mode)
        self.bits = bits
        self.fallback_gelu = nn.GELU()

    def forward(self, x, scaling_factor: Optional[torch.Tensor] = None):
        # Handle tuple input `(tensor, sf)` coming from previous QuantLinear
        if isinstance(x, tuple):
            x, scaling_factor = x  # unpack

        if self.int_gelu.quant_mode == "none" or scaling_factor is None:
            # Float fallback (Phase 1 compatibility)
            return self.fallback_gelu(x)

        # Integer-only path
        return self.int_gelu(x, scaling_factor)

class QuantizedViTAttention(nn.Module):
    def __init__(self, attn: nn.Module, qconf: dict):
        super().__init__()
        self.config = attn.config
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        quant_mode = qconf["quant_mode"]
        weight_bit = qconf["bits"]  # Use consistent bit setting
        bias_bit = 32

        # CRITICAL FIX 1: Initialize input quantizer to provide scaling factors
        self.input_act = QuantAct(8, quant_mode=quant_mode)

        # Projections → QuantLinear
        self.q_proj = QuantLinear(weight_bit, bias_bit, per_channel=True, quant_mode=quant_mode)
        self.k_proj = QuantLinear(weight_bit, bias_bit, per_channel=True, quant_mode=quant_mode)
        self.v_proj = QuantLinear(weight_bit, bias_bit, per_channel=True, quant_mode=quant_mode)
        
        # CRITICAL FIX 2: Properly set parameters to actually quantize weights
        self.q_proj.set_param(attn.query)
        self.k_proj.set_param(attn.key)
        self.v_proj.set_param(attn.value)

        # Post-proj activation quantizers
        self.q_proj_act = QuantAct(8, quant_mode=quant_mode)
        self.k_proj_act = QuantAct(8, quant_mode=quant_mode)
        self.v_proj_act = QuantAct(8, quant_mode=quant_mode)

        self.softmax = IntSoftmax(8, quant_mode=quant_mode)
        self.attn_probs_act = QuantAct(8, quant_mode=quant_mode)
        self.attn_out_act = QuantAct(8, quant_mode=quant_mode)

        dropout_prob = getattr(attn, "dropout", 0.0)
        if hasattr(dropout_prob, "p"):
            dropout_prob = dropout_prob.p
        self.dropout = nn.Dropout(float(dropout_prob))

        if hasattr(attn, "out_proj"):
            self.out_proj = QuantLinear(weight_bit, bias_bit, per_channel=True, quant_mode=quant_mode)
            self.out_proj.set_param(attn.out_proj)
        else:
            self.out_proj = None

    def _transpose(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # CRITICAL FIX 3: Get input scaling factor for proper quantization chain
        hidden_states_2d = hidden_states.view(-1, hidden_size)
        
        # Quantize input and get scaling factor
        input_quantized, input_scaling_factor = self.input_act(hidden_states_2d)
        
        # CRITICAL FIX 4: Pass scaling factor to QuantLinear to enable integer path
        q_out = self.q_proj(input_quantized, input_scaling_factor)
        k_out = self.k_proj(input_quantized, input_scaling_factor)
        v_out = self.v_proj(input_quantized, input_scaling_factor)
        
        # Handle tuple outputs from QuantLinear
        if isinstance(q_out, tuple):
            q, q_sf = q_out
        else:
            q, q_sf = q_out, None
            
        if isinstance(k_out, tuple):
            k, k_sf = k_out
        else:
            k, k_sf = k_out, None
            
        if isinstance(v_out, tuple):
            v, v_sf = v_out
        else:
            v, v_sf = v_out, None
        
        # Reshape back to 3D
        q = q.view(batch_size, seq_len, -1)
        k = k.view(batch_size, seq_len, -1)
        v = v.view(batch_size, seq_len, -1)
        
        # CRITICAL FIX 5: Maintain scaling factor chain
        q, q_sf = self.q_proj_act(q, q_sf)
        k, k_sf = self.k_proj_act(k, k_sf)
        v, v_sf = self.v_proj_act(v, v_sf)

        # Apply attention scaling and transpose
        q = self._transpose(q * self.scaling)
        k = self._transpose(k)
        v = self._transpose(v)

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        
        # Use quantized softmax with scaling factor
        attn_probs = self.softmax(attn_weights, scaling_factor=q_sf)
        
        if head_mask is not None:
            attn_probs = attn_probs * head_mask
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.all_head_size)
        
        context, context_sf = self.attn_out_act(context)
        
        if self.out_proj is not None:
            context_2d = context.view(-1, self.all_head_size)
            output = self.out_proj(context_2d, context_sf)
            if isinstance(output, tuple):
                output = output[0]
            output = output.view(batch_size, seq_len, -1)
        else:
            output = context

        if output_attentions:
            return output, attn_probs
        return (output,)


class QuantizedViTOutput(nn.Module):
    def __init__(self, out_block: nn.Module, qconf: dict):
        super().__init__()
        quant_mode = qconf["quant_mode"]
        weight_bit = qconf["bits"]  # Use consistent bit setting
        
        self.dense = QuantLinear(weight_bit, 32, per_channel=True, quant_mode=quant_mode)
        self.dense.set_param(out_block.dense)
        
        raw_dropout = getattr(out_block, "dropout", 0.0)
        if isinstance(raw_dropout, nn.Dropout):
            p = raw_dropout.p
        else:
            p = float(raw_dropout)
        self.dropout = nn.Dropout(p)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        hidden_states_2d = hidden_states.view(-1, hidden_size)
        
        # CRITICAL: Need to handle scaling factor properly
        if isinstance(hidden_states_2d, tuple):
            hidden_states_2d, sf = hidden_states_2d
            dense_out = self.dense(hidden_states_2d, sf)
        else:
            dense_out = self.dense(hidden_states_2d)
            
        if isinstance(dense_out, tuple):
            dense_out = dense_out[0]
            
        hidden_states = dense_out.view(batch_size, seq_len, -1)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


class ViTQuantizer:
    def __init__(
        self,
        *,
        bits: int = 8,
        method: str = "tensor",
        p: float = 1.0,
        quantize_attention: bool = True,
        quantize_mlp: bool = True,
        quantize_classifier: bool = True,
        quantize_nonlinear: bool = True,
        add_quant_act_modules: bool = True,
    ) -> None:
        self.bits = bits
        self.method = method
        self.p = p
        self.quantize_attention = quantize_attention
        self.quantize_mlp = quantize_mlp
        self.quantize_classifier = quantize_classifier
        self.quantize_nonlinear = quantize_nonlinear
        self.add_quant_act_modules = add_quant_act_modules

    # Map method to quant_mode
        quant_mode = "symmetric" if method == "tensor" else "histogram"
        self.quant_config = {
            "bits": bits,
            "method": method,
            "p": p,
            "quant_mode": quant_mode,
    }

        self.int_linear = 0
        self.int_gelu = 0
        self.int_ln = 0
        self.int_softmax = 0
        self.qact = 0

    def _replace_linear(self, layer: nn.Linear) -> QuantLinear:
        ql = QuantLinear(self.bits, 32, per_channel=True, quant_mode=self.quant_config["quant_mode"])
        ql.set_param(layer)
        self.int_linear += 1
        return ql

    def _patch_layernorm(self, ln: nn.LayerNorm) -> IntLayerNorm:
        iln = IntLayerNorm(32, quant_mode=self.quant_config["quant_mode"])
        iln.set_param(ln)
        self.int_ln += 1
        return iln

    def remove_activation_quantizers(self):
        pass

    def _quantize_encoder_layer(self, layer) -> None:
        if self.quantize_attention:
            original_attn = layer.attention.attention
            layer.attention.attention = QuantizedViTAttention(original_attn, self.quant_config)
            self.int_softmax += 1
            layer.attention.output = QuantizedViTOutput(layer.attention.output, self.quant_config)
        
        if self.quantize_mlp:
            layer.intermediate.dense = self._replace_linear(layer.intermediate.dense)
            layer.output.dense = self._replace_linear(layer.output.dense)
            
        # GELU ----------------------------------------------------------------------
        if self.quantize_nonlinear:
            layer.intermediate.intermediate_act_fn = IntGELUWrapper(
                quant_mode=self.quant_config["quant_mode"], bits=self.bits
            )
            self.int_gelu += 1

        if self.quantize_nonlinear:
            if hasattr(layer, "layernorm_before"):
                layer.layernorm_before = self._patch_layernorm(layer.layernorm_before)
            if hasattr(layer, "layernorm_after"):
                layer.layernorm_after = self._patch_layernorm(layer.layernorm_after)

        if self.add_quant_act_modules and not hasattr(layer, "mlp_output_act"):
            layer.mlp_output_act = QuantAct(8, quant_mode=self.quant_config["quant_mode"])
            self.qact += 1

    def quantize_model(
        self, model: Union[ViTModel, ViTForImageClassification]
    ) -> Union[ViTModel, ViTForImageClassification]:
        q_model = copy.deepcopy(model)
        
        # CRITICAL FIX 7: Add debug verification
        print(f"Quantizing model with {self.bits} bits, mode: {self.quant_config['quant_mode']}")
        
        for i, enc_layer in enumerate(q_model.vit.encoder.layer):
            self._quantize_encoder_layer(enc_layer)
            print(f"Quantized encoder layer {i}")

        if self.quantize_classifier and hasattr(q_model, "classifier"):
            original_weight = q_model.classifier.weight.data.clone()
            q_model.classifier = self._replace_linear(q_model.classifier)
            quantized_weight = q_model.classifier.weight.data
            weight_diff = torch.mean((original_weight - quantized_weight) ** 2).item()
            print(f"Classifier weight MSE after quantization: {weight_diff:.6f}")

        logger.info(
            "Quantisation summary → IntLinear %d, IntGELU %d, IntLayerNorm %d, IntSoftmax %d, QuantAct %d",
            self.int_linear,
            self.int_gelu,
            self.int_ln,
            self.int_softmax,
            self.qact,
        )
        return q_model

    def debug_quantization_status(self, model):
        """Add this method to verify quantization is working"""
        print("\n=== QUANTIZATION DEBUG ===")
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, 'weight') and hasattr(module, 'weight_integer'):
                    weight_diff = torch.mean((module.weight - module.weight_integer) ** 2).item()
                    print(f"{name}: QuantLinear, weight MSE: {weight_diff:.6f}, mode: {module.quant_mode}")
                else:
                    print(f"{name}: QuantLinear, mode: {module.quant_mode}")
            elif isinstance(module, QuantAct):
                print(f"{name}: QuantAct, mode: {module.quant_mode}")
        print("========================\n")


# -----------------------------------------------------------------------------
# Utility functions for testing
# -----------------------------------------------------------------------------

def compare_models(original_model, quantized_model, processor, inputs):
    """Compare original and quantized models"""
    try:
        original_model.eval()
        quantized_model.eval()
        
        with torch.no_grad():
            # Get outputs from both models
            original_outputs = original_model(**inputs)
            quantized_outputs = quantized_model(**inputs)
            
            original_logits = original_outputs.logits
            quantized_logits = quantized_outputs.logits
            
            # Calculate metrics
            mse = torch.mean((original_logits - quantized_logits) ** 2).item()
            mae = torch.mean(torch.abs(original_logits - quantized_logits)).item()
            
            # Top-1 accuracy match
            original_pred = original_logits.argmax(-1)
            quantized_pred = quantized_logits.argmax(-1)
            top1_match = (original_pred == quantized_pred).float().mean().item()
            
            return {
                'mse': mse,
                'mae': mae,
                'top1_accuracy_match': top1_match,
                'original_logits': original_logits,
                'quantized_logits': quantized_logits
            }
    except Exception as e:
        print(f"Error in model comparison: {e}")
        return None

# Utility: addition with scaling-factor alignment ---------------------------------

def add_with_scaling(
    x: torch.Tensor,
    x_sf: Optional[torch.Tensor],
    y: torch.Tensor,
    y_sf: Optional[torch.Tensor],
):
    """Add two tensors that might have different scaling factors.

    If either scaling factor is None we fall back to float addition (Phase-1).
    Otherwise we up-/down-scale the integer representations so that they share
    the same scale, following I-BERT Section 3.2.
    Returns (tensor, scaling_factor).
    """
    if x_sf is None or y_sf is None:
        # Float fallback – behave exactly like PyTorch add
        return x + y, None

    # Ensure both scaling factors are scalar tensors on same device
    if not torch.is_tensor(x_sf):
        x_sf = torch.tensor(x_sf, device=x.device, dtype=x.dtype)
    if not torch.is_tensor(y_sf):
        y_sf = torch.tensor(y_sf, device=y.device, dtype=y.dtype)

    # Choose the larger scaling factor to minimise precision loss
    common_sf = torch.max(x_sf, y_sf)
    x_int = x / x_sf
    y_int = y / y_sf
    # Rescale to common scale
    x_int = x_int * (x_sf / common_sf)
    y_int = y_int * (y_sf / common_sf)

    out_int = x_int + y_int
    out = out_int * common_sf
    return out, common_sf