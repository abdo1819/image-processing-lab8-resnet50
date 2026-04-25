"""
Lab 8 – Part 2: ResNet50 Fine-Tuning (Classifier-Only)
=======================================================
Strategy
--------
  • Load ResNet50 with ImageNet weights.
  • FREEZE every layer except the final fully-connected head.
  • Replace the head with a 4-class output layer.
  • Train only that head on our dataset (cats / dogs / horses / Humans).
  • Save the model and plot training curves:
      output/training_curves_part2.png
      trained_models/resnet50_finetuned.pth

Offline mode: set TORCH_HOME so PyTorch reads cached weights.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(SCRIPT_DIR, "data", "data")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "output")
MODELS_OUT     = os.path.join(SCRIPT_DIR, "trained_models")
OFFLINE_MODELS = os.path.join(SCRIPT_DIR, "offline_packages", "models")

os.environ["TORCH_HOME"] = OFFLINE_MODELS
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_OUT,  exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_CLASSES = 4
BATCH_SIZE  = 16
NUM_EPOCHS  = 10
LR          = 1e-3
VAL_SPLIT   = 0.2    # 80 % train / 20 % val
SEED        = 42

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Dataset ────────────────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_ds = ImageFolder(DATA_DIR, transform=tf)
print(f"Classes: {full_ds.classes}  |  Total images: {len(full_ds)}")

n_val   = int(len(full_ds) * VAL_SPLIT)
n_train = len(full_ds) - n_val
train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

# ── 2. Model – freeze all, replace head ──────────────────────────────────────
print("\nLoading ResNet50 (ImageNet weights)…")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

# ── 3. Loss / optimizer ───────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# ── 4. Training loop ──────────────────────────────────────────────────────────
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print()
for epoch in range(NUM_EPOCHS):
    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}]")

    for phase in ("train", "val"):
        model.train() if phase == "train" else model.eval()
        loader = train_loader if phase == "train" else val_loader

        total_loss, correct = 0.0, 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs      = model(inputs)
                loss         = criterion(outputs, labels)
                _, preds     = torch.max(outputs, 1)
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct    += (preds == labels).sum().item()

        epoch_loss = total_loss / len(loader.dataset)
        epoch_acc  = correct   / len(loader.dataset)
        history[f"{phase}_loss"].append(epoch_loss)
        history[f"{phase}_acc"].append(epoch_acc)
        print(f"  {phase:5s}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

# ── 5. Save model ─────────────────────────────────────────────────────────────
save_path = os.path.join(MODELS_OUT, "resnet50_finetuned.pth")
torch.save(model.state_dict(), save_path)
print(f"\nModel saved → {save_path}")

# ── 6. Training curves ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_x  = range(1, NUM_EPOCHS + 1)

axes[0].plot(epochs_x, history["train_loss"], "o-", label="Train")
axes[0].plot(epochs_x, history["val_loss"],   "s-", label="Val")
axes[0].set_title("Loss"),   axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss"),  axes[0].legend(), axes[0].grid(True)

axes[1].plot(epochs_x, history["train_acc"], "o-", label="Train")
axes[1].plot(epochs_x, history["val_acc"],   "s-", label="Val")
axes[1].set_title("Accuracy"), axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy"), axes[1].legend(), axes[1].grid(True)

plt.suptitle("ResNet50 Fine-Tuning – Training Curves", fontsize=13)
plt.tight_layout()
curves_path = os.path.join(OUTPUT_DIR, "training_curves_part2.png")
plt.savefig(curves_path, dpi=150)
print(f"Training curves → {curves_path}")
