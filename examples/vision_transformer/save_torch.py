import torch
from fairseq.models.vit_quantization import ViTQuantizer
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
quantizer = ViTQuantizer(bits=8, method="tensor", p=1.0)
quantized_model = quantizer.quantize_model(model)

torch.save(quantized_model.state_dict(), "vit_int8_model.pt")