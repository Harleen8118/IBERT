# Vision Transformer (ViT) Quantization Examples

This directory contains examples and utilities for quantizing Vision Transformer models using I-BERT's integer-only quantization framework.

## Overview

We've extended I-BERT's quantization capabilities from BERT/RoBERTa models to Vision Transformers, supporting:

- **4-bit and 8-bit quantization** for ViT models
- **Per-tensor, per-channel, and histogram-based** quantization methods
- **Attention, MLP, and classifier head** quantization
- **CPU and CUDA** support

## Files

| File | Description |
|------|-------------|
| `test_ibert.py` | Basic I-BERT functionality test (CPU-only) |
| `test_vit_quantization.py` | ViT model exploration and basic quantization test |
| `test_with_image.py` | **Main demo**: End-to-end ViT quantization with sample image |
| `sample_image.jpg` | Sample image for testing |

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
uv venv vit-quant-env
source vit-quant-env/bin/activate  # Linux/macOS
# or: .\vit-quant-env\Scripts\activate  # Windows

# Install dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install transformers pillow requests
uv pip install -e ../..  # Install I-BERT in editable mode
```

### 2. Run Basic Tests

```bash
# Test I-BERT installation
python test_ibert.py

# Test ViT model loading and basic quantization
python test_vit_quantization.py

# Run full ViT quantization demo
python test_with_image.py
```

### 3. Use in Your Code

```python
from fairseq.models.vit_quantization import ViTQuantizer
from transformers import ViTForImageClassification, ViTImageProcessor

# Load model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Quantize to 4-bit
quantizer = ViTQuantizer(
    bits=4, 
    method="tensor", 
    p=1.0,
    quantize_attention=True,
    quantize_mlp=True,
    quantize_classifier=True
)
quantized_model = quantizer.quantize_model(model)

# Use quantized model for inference
inputs = processor(images=your_image, return_tensors="pt")
with torch.no_grad():
    outputs = quantized_model(**inputs)
    predictions = outputs.logits.softmax(-1)
```

## Quantization Options

### Bit Widths
- `bits=8`: 8-bit quantization (default)
- `bits=4`: 4-bit quantization

### Methods
- `method="tensor"`: Per-tensor quantization (fastest)
- `method="channel"`: Per-channel quantization (better accuracy)
- `method="histogram"`: Histogram-based quantization (best accuracy)

### Components
- `quantize_attention=True`: Quantize attention layers
- `quantize_mlp=True`: Quantize MLP/feed-forward layers  
- `quantize_classifier=True`: Quantize final classifier head

### Quantization Probability
- `p=1.0`: Quantize all selected layers (default)
- `p=0.5`: Quantize 50% of selected layers (for gradual quantization)

## Expected Results

With the sample image, you should see results like:

```
Original prediction: microphone, mike (confidence ≈0.94)

--- Testing 4-bit tensor quantization ---
Quantized 169 linear layers
MSE Loss: 0.011xxx
MAE Loss: 0.008xxx  
Top-1 Match: 100% ✅
```

## Performance Notes

- **4-bit quantization** typically provides 2-4x memory reduction with minimal accuracy loss
- **Per-tensor** quantization is fastest but may have slightly lower accuracy
- **Histogram** quantization provides best accuracy but is slower to compute
- **CUDA** acceleration is used automatically when available

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use CPU-only mode or smaller batch sizes
2. **Import errors**: Ensure I-BERT is installed in editable mode: `pip install -e ../..`
3. **Numpy compatibility**: The code handles numpy deprecation warnings automatically

### CPU-Only Mode

All examples work on CPU-only systems. CUDA acceleration is used automatically when available.

## Further Reading

See `../../docs/vision_transformer_quantization.rst` for detailed documentation and advanced usage patterns. 