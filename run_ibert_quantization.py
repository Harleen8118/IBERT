import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
import copy
import math
import sys

# --- I-BERT Quantization Imports ---
# Add the fairseq path to the python path to allow imports
# Make sure this path is correct for your environment
sys.path.insert(0, 'c:\\Users\\klinu\\Desktop\\I-BERT')
from fairseq.quantization.utils.quant_modules import (
    IntGELU, IntLayerNorm, QuantAct, IntSoftmax, QuantLinear
)
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTOutput, ViTIntermediate
from transformers.activations import GELUActivation

# --- Configuration ---
MODEL_NAME = "google/vit-base-patch16-224"
DATASET_PATH = "./data"
NUM_LABELS = 10  # CIFAR-10
FP32_FINETUNED_PATH = "models/vit_cifar10_finetuned.pt"
QAT_MODEL_PATH = "models/vit_cifar10_qat.pt"

# --- Quantization Wrapper Classes ---

class QuantGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_act = QuantAct(activation_bit=8, quant_mode='symmetric', running_stat=True)
        self.int_gelu = IntGELU(quant_mode='symmetric')

    def forward(self, x):
        quant_x, scaling_factor = self.quant_act(x)
        output = self.int_gelu(quant_x, scaling_factor)
        return output

class QuantSoftmax(nn.Module):
    def __init__(self, output_bit=8):
        super().__init__()
        self.quant_act = QuantAct(activation_bit=16, quant_mode='symmetric', running_stat=True)
        self.int_softmax = IntSoftmax(output_bit=output_bit, quant_mode='symmetric')

    def forward(self, x):
        quant_x, scaling_factor = self.quant_act(x)
        return self.int_softmax(quant_x, scaling_factor)

class QuantLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps):
        super().__init__()
        self.int_layernorm = IntLayerNorm(output_bit=8, quant_mode='symmetric', normalized_shape=normalized_shape, eps=eps)
        self.quant_act = QuantAct(activation_bit=8, quant_mode='symmetric', running_stat=True)

    def forward(self, x):
        quant_x, scaling_factor = self.quant_act(x)
        return self.int_layernorm(quant_x, scaling_factor)

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.quant_act = QuantAct(activation_bit=8, quant_mode='symmetric', running_stat=True)
        self.quant_linear = QuantLinear(weight_bit=8, bias_bit=32, in_features=in_features, out_features=out_features)
        if not bias:
            self.quant_linear.bias = None

    def forward(self, x):
        quant_x, act_scaling_factor = self.quant_act(x)
        output, _ = self.quant_linear(quant_x, act_scaling_factor)
        return output

class CustomViTSelfAttentionWithQuant(ViTSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.quant_softmax = QuantSoftmax(output_bit=8)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
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

# --- Helper Functions ---

def get_cifar10_dataloaders(image_processor, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((image_processor.size['height'], image_processor.size['width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])
    train_dataset = datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return total_loss / len(dataloader), accuracy

def finetune_fp32_model(model, train_loader, test_loader, device, epochs=3):
    print("--- Starting FP32 Model Fine-tuning ---")
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    torch.save(model.state_dict(), FP32_FINETUNED_PATH)
    print(f"FP32 model fine-tuned and saved to {FP32_FINETUNED_PATH}")

def prepare_model_for_quantization(model):
    print("--- Preparing Model for I-BERT Quantization ---")
    model_quant = copy.deepcopy(model)
    for name, module in model_quant.named_modules():
        if isinstance(module, GELUActivation):
            setattr(model_quant.get_submodule(name.rsplit('.', 1)[0]), name.rsplit('.', 1)[1], QuantGELU())
        elif isinstance(module, nn.LayerNorm):
            setattr(model_quant.get_submodule(name.rsplit('.', 1)[0]), name.rsplit('.', 1)[1], QuantLayerNorm(module.normalized_shape, module.eps))
        elif isinstance(module, ViTSelfAttention):
             setattr(model_quant.get_submodule(name.rsplit('.', 1)[0]), name.rsplit('.', 1)[1], CustomViTSelfAttentionWithQuant(model.config))
        elif isinstance(module, nn.Linear) and name != 'classifier':
            setattr(model_quant.get_submodule(name.rsplit('.', 1)[0]), name.rsplit('.', 1)[1], QuantLinear(module.in_features, module.out_features, module.bias is not None))
    return model_quant

def calibrate_model(model, dataloader, device):
    print("--- Calibrating Model for Static Quantization ---")
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Calibrating"):
            model(pixel_values=images.to(device))

    # Set running_stat to False for all QuantAct modules after calibration
    for module in model.modules():
        if isinstance(module, QuantAct):
            module.running_stat = False
    print("Calibration finished and quantization ranges frozen.")

def quantization_aware_training(model, train_loader, test_loader, device, epochs=3):
    print("--- Starting Quantization-Aware Training (QAT) ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # Use a smaller learning rate for QAT
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"QAT Epoch {epoch+1}/{epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    torch.save(model.state_dict(), QAT_MODEL_PATH)
    print(f"QAT model trained and saved to {QAT_MODEL_PATH}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model and processor
    model = ViTForImageClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
    model.to(device)
    image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    # 2. Get Dataloaders
    train_loader, test_loader = get_cifar10_dataloaders(image_processor)

    # 3. FP32 Fine-tuning
    finetune_fp32_model(model, train_loader, test_loader, device, epochs=3)

    # 4. Prepare for Quantization
    model.load_state_dict(torch.load(FP32_FINETUNED_PATH))
    model_quant = prepare_model_for_quantization(model)
    model_quant.to(device)

    # 5. Calibration
    # Use a subset of the training data for calibration
    calibration_loader = DataLoader(list(train_loader.dataset)[:500], batch_size=32)
    calibrate_model(model_quant, calibration_loader, device)

    # 6. Quantization-Aware Training
    quantization_aware_training(model_quant, train_loader, test_loader, device, epochs=3)

    # 7. Final Evaluation
    print("--- Final Evaluation of QAT Model ---")
    criterion = nn.CrossEntropyLoss()
    _, final_accuracy = evaluate(model_quant, test_loader, criterion, device)
    print(f"Final Quantized Model Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()
