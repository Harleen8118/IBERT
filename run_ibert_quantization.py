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
import os

# --- I-BERT Quantization Imports ---
sys.path.insert(0, 'c:\\Users\\klinu\\Desktop\\I-BERT')
from fairseq.quantization.utils.quant_modules import (
    IntGELU, IntLayerNorm, QuantAct, IntSoftmax, QuantLinear as QuantLinearBase
)
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTOutput, ViTIntermediate

# --- Configuration ---
MODEL_NAME = "google/vit-base-patch16-224"
DATASET_PATH = "./data"
NUM_LABELS = 10  # CIFAR-10
FP32_FINETUNED_PATH = "models/vit_cifar10_finetuned.pt"
CALIBRATED_MODEL_PATH = "models/vit_cifar10_calibrated.pt"
QAT_MODEL_PATH = "models/vit_cifar10_qat.pt"
 
# --- Robust I-BERT Quantization Modules (inspired by vit_quantization.py) ---

class QuantizedViTAttention(nn.Module):
    def __init__(self, attn_module, dropout_prob, quant_mode='symmetric', weight_bit=8):
        super().__init__()
        self.attention = attn_module
        self.num_attention_heads = attn_module.num_attention_heads
        self.attention_head_size = attn_module.attention_head_size
        self.all_head_size = attn_module.all_head_size

        self.query = QuantLinearBase(weight_bit=weight_bit, bias_bit=32, quant_mode=quant_mode)
        self.key = QuantLinearBase(weight_bit=weight_bit, bias_bit=32, quant_mode=quant_mode)
        self.value = QuantLinearBase(weight_bit=weight_bit, bias_bit=32, quant_mode=quant_mode)
        self.query.set_param(attn_module.query)
        self.key.set_param(attn_module.key)
        self.value.set_param(attn_module.value)

        self.dropout = nn.Dropout(dropout_prob)
        self.quant_softmax = IntSoftmax(output_bit=8, quant_mode=quant_mode)
        self.act_quant = QuantAct(activation_bit=8, quant_mode=quant_mode, running_stat=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        quant_hidden_states, sf = self.act_quant(hidden_states)

        query_layer, _ = self.query(quant_hidden_states, sf)
        key_layer, _ = self.key(quant_hidden_states, sf)
        value_layer, _ = self.value(quant_hidden_states, sf)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.quant_softmax(attention_scores, sf)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class QuantizedViTOutput(nn.Module):
    def __init__(self, output_module, quant_mode='symmetric', weight_bit=8):
        super().__init__()
        self.dense = QuantLinearBase(weight_bit=weight_bit, bias_bit=32, quant_mode=quant_mode)
        self.dense.set_param(output_module.dense)
        self.dropout = output_module.dropout
        self.act_quant = QuantAct(activation_bit=8, quant_mode=quant_mode, running_stat=True)

    def forward(self, hidden_states, input_tensor):
        quant_hidden_states, sf = self.act_quant(hidden_states)
        hidden_states, _ = self.dense(quant_hidden_states, sf)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor

class ViTQuantizer:
    def __init__(self, bits=8, quant_mode='symmetric'):
        self.bits = bits
        self.quant_mode = quant_mode

    def _replace_linear(self, module):
        ql = QuantLinearBase(weight_bit=self.bits, bias_bit=32, quant_mode=self.quant_mode)
        ql.set_param(module)
        return ql

    def quantize(self, model):
        print("--- Starting Model Quantization ---")
        q_model = copy.deepcopy(model)

        dropout_prob = model.config.attention_probs_dropout_prob
        for layer in q_model.vit.encoder.layer:
            # Quantize self-attention
            layer.attention.attention = QuantizedViTAttention(
                layer.attention.attention, 
                dropout_prob, 
                quant_mode=self.quant_mode, 
                weight_bit=self.bits
            )
            # Quantize self-attention output
            layer.attention.output = QuantizedViTOutput(
                layer.attention.output, 
                quant_mode=self.quant_mode, 
                weight_bit=self.bits
            )
            # Quantize intermediate layer (MLP)
            layer.intermediate.dense = self._replace_linear(layer.intermediate.dense)
            # Quantize output layer (MLP)
            layer.output.dense = self._replace_linear(layer.output.dense)
            # Quantize GELU
            layer.intermediate.intermediate_act_fn = IntGELU(quant_mode=self.quant_mode)
            # Quantize LayerNorms
            layer.layernorm_before = IntLayerNorm(
                normalized_shape=layer.layernorm_before.normalized_shape, 
                eps=layer.layernorm_before.eps, 
                quant_mode=self.quant_mode
            )
            layer.layernorm_after = IntLayerNorm(
                normalized_shape=layer.layernorm_after.normalized_shape, 
                eps=layer.layernorm_after.eps, 
                quant_mode=self.quant_mode
            )

        # Quantize the final classifier
        q_model.classifier = self._replace_linear(q_model.classifier)
        print("--- Model Quantization Finished ---")
        return q_model

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
    os.makedirs(os.path.dirname(FP32_FINETUNED_PATH), exist_ok=True)
    torch.save(model.state_dict(), FP32_FINETUNED_PATH)
    print(f"FP32 model fine-tuned and saved to {FP32_FINETUNED_PATH}")

def calibrate_model(model, dataloader, device):
    print("--- Calibrating Model for Static Quantization ---")
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Calibrating"):
            model(pixel_values=images.to(device))

    for module in model.modules():
        if isinstance(module, QuantAct):
            module.running_stat = False
    print("Calibration finished and quantization ranges frozen.")
    os.makedirs(os.path.dirname(CALIBRATED_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), CALIBRATED_MODEL_PATH)
    print(f"Calibrated model saved to {CALIBRATED_MODEL_PATH}")

def quantization_aware_training(model, train_loader, test_loader, device, epochs=3):
    print("--- Starting Quantization-Aware Training (QAT) ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"QAT Epoch {epoch+1}/{epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    os.makedirs(os.path.dirname(QAT_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), QAT_MODEL_PATH)
    print(f"QAT model trained and saved to {QAT_MODEL_PATH}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ViTForImageClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
    model.to(device)
    image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    train_loader, test_loader = get_cifar10_dataloaders(image_processor)

    # Step 1: Fine-tune the FP32 model
    finetune_fp32_model(model, train_loader, test_loader, device, epochs=3)

    # Step 2: Quantize the fine-tuned model
    model.load_state_dict(torch.load(FP32_FINETUNED_PATH))
    quantizer = ViTQuantizer(quant_mode='symmetric', bits=8)
    model_quant = quantizer.quantize(model)
    model_quant.to(device)

    # Step 3: Calibrate the quantized model
    calibration_loader = DataLoader(list(train_loader.dataset)[:500], batch_size=32, shuffle=False)
    calibrate_model(model_quant, calibration_loader, device)

    # Step 4: Quantization-Aware Training
    quantization_aware_training(model_quant, train_loader, test_loader, device, epochs=3)

    # Step 5: Final Evaluation
    print("--- Final Evaluation of QAT Model ---")
    criterion = nn.CrossEntropyLoss()
    _, final_accuracy = evaluate(model_quant, test_loader, criterion, device)
    print(f"Final Quantized Model Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()
