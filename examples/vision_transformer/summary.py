import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from torchinfo import summary

# Load the model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Print model architecture
print(model)

# Dummy input (batch_size=1, 3-channel 224x224 image)
dummy_input = torch.randn(1, 3, 224, 224)

# Get summary
summary(
    model,
    input_data=dummy_input,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    depth=4,
)
