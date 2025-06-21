"""
import onnx

def check_onnx_quantization(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    quantized_ops = set()
    unquantized_ops = set()

    for node in graph.node:
        op_type = node.op_type
        if "QuantizeLinear" in op_type or "DequantizeLinear" in op_type or "QLinear" in op_type:
            quantized_ops.add(op_type)
        elif op_type in ["Gemm", "MatMul", "Add", "Relu", "Gelu", "Softmax", "LayerNormalization"]:
            unquantized_ops.add(op_type)

    print("✅ Quantized ops found:")
    for op in sorted(quantized_ops):
        print(f"  {op}")
    
    print("\n🚨 FP32 ops likely present:")
    for op in sorted(unquantized_ops):
        print(f"  ❌ {op}")
    
    return len(unquantized_ops) == 0

if __name__ == "__main__":
    check_onnx_quantization("vit_int8_model.onnx")
"""
import torch
import torch.nn as nn
from fairseq.modules.quantization.scalar.modules import IntLinear, IntGELU, IntLayerNorm, IntSoftmax, ActivationQuantizer

def check_full_quantization(model):
    print("\n🔍 Checking full quantization coverage...\n")

    fp32_ops = []
    quant_ops = []

    for name, module in model.named_modules():
        # Skip known fallbacks that are unused
        if "fallback_gelu" in name:
            continue

        # Detect unquantized modules
        if isinstance(module, (nn.Linear, nn.GELU, nn.LayerNorm, nn.Softmax)):
            fp32_ops.append((name, type(module).__name__))

        # Detect quantized modules
        elif isinstance(module, (IntLinear, IntGELU, IntLayerNorm, IntSoftmax, ActivationQuantizer)):
            quant_ops.append((name, type(module).__name__))

    # Print quantized modules
    print("✅ Quantized Modules:")
    for name, typ in quant_ops:
        print(f"  ✔️ {name}: {typ}")

    # Print unquantized modules
    print("\n🚨 Unquantized FP32 Modules:")
    if fp32_ops:
        for name, typ in fp32_ops:
            print(f"  ❌ {name}: {typ}")
    else:
        print("  🎉 None found — all layers quantized!")

    # Print summary
    print("\n📊 Summary:")
    print(f"  ➤ Quantized: {len(quant_ops)}")
    print(f"  ➤ Unquantized: {len(fp32_ops)}")
    print(f"\n✅ Status: {'PASS' if len(fp32_ops) == 0 else 'FAIL'}")
    return len(fp32_ops) == 0


if __name__ == "__main__":
    # Load the full quantized model
    from fairseq.models.vit_quantization import ViTQuantizer
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    quantizer = ViTQuantizer(bits=8, method="tensor", p=1.0)
    quantized_model = quantizer.quantize_model(model)

    check_full_quantization(quantized_model)
