"""
Lab 8 – Part 2: ResNet50 Fine-Tuning (Classifier-Only)
=======================================================
Strategy
--------
  • Load ResNet50 with ImageNet weights.
  • FREEZE every layer except the final fully-connected head.
  • Replace the head with a 4-class output layer.
  • Train only that head on our dataset (cats / dogs / horses / Humans).
  • Evaluate on a held-out validation split and generate:
      output/training_curves_part2.png
      output/confusion_matrix_part2.png
      output/classification_report_part2.txt
      trained_models/resnet50_finetuned.pth

Offline mode: set TORCH_HOME so PyTorch reads cached weights.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(SCRIPT_DIR, "data", "data")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "output")
MODELS_OUT     = os.path.join(SCRIPT_DIR, "trained_models")
OFFLINE_MODELS = os.path.join(SCRIPT_DIR, "offline_packages", "models")

os.environ["TORCH_HOME"] = OFFLINE_MODELS   # use local cache (online-safe too)
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

# ── 1. Transforms ─────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── 2. Dataset with per-split transforms ─────────────────────────────────────
class SplitDataset(torch.utils.data.Dataset):
    """Wraps an ImageFolder, applies a given transform to a subset of indices."""
    def __init__(self, folder: ImageFolder, indices: list, transform):
        self.folder    = folder
        self.indices   = indices
        self.transform = transform
        self.loader    = default_loader

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.folder.imgs[self.indices[idx]]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


full_ds     = ImageFolder(DATA_DIR)                 # scan folder structure
class_names = full_ds.classes
print(f"Classes: {class_names}  |  Total images: {len(full_ds)}")

# Deterministic train/val split
n_total = len(full_ds)
n_val   = int(n_total * VAL_SPLIT)
n_train = n_total - n_val
gen     = torch.Generator().manual_seed(SEED)
train_idx, val_idx = torch.utils.data.random_split(
    range(n_total), [n_train, n_val], generator=gen
)

train_ds = SplitDataset(full_ds, list(train_idx), train_tf)
val_ds   = SplitDataset(full_ds, list(val_idx),   val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=False)

print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

# ── 3. Model – freeze all, replace head ──────────────────────────────────────
print("\nLoading ResNet50 (ImageNet weights)…")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze every parameter
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier head (only this layer will be trained)
in_features = model.fc.in_features          # 2048 for ResNet50
model.fc    = nn.Linear(in_features, NUM_CLASSES)
# model.fc parameters are requires_grad=True by default (newly created layer)

model.to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}  "
      f"({100*trainable/total:.2f}%)")

# ── 4. Loss / optimizer / scheduler ──────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ── 5. Training loop ──────────────────────────────────────────────────────────
best_weights = copy.deepcopy(model.state_dict())
best_acc     = 0.0
history      = {"train_loss": [], "train_acc": [],
                "val_loss":   [], "val_acc":   []}

print()
for epoch in range(NUM_EPOCHS):
    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}]  lr={scheduler.get_last_lr()[0]:.1e}")

    for phase in ("train", "val"):
        model.train() if phase == "train" else model.eval()
        loader = train_loader if phase == "train" else val_loader

        running_loss = 0.0
        running_correct = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs         = model(inputs)
                loss            = criterion(outputs, labels)
                _, preds        = torch.max(outputs, 1)
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss    += loss.item() * inputs.size(0)
            running_correct += (preds == labels).sum().item()

        if phase == "train":
            scheduler.step()

        epoch_loss = running_loss    / len(loader.dataset)
        epoch_acc  = running_correct / len(loader.dataset)
        history[f"{phase}_loss"].append(epoch_loss)
        history[f"{phase}_acc"].append(epoch_acc)
        print(f"  {phase:5s}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

        if phase == "val" and epoch_acc > best_acc:
            best_acc     = epoch_acc
            best_weights = copy.deepcopy(model.state_dict())

print(f"\nBest validation accuracy: {best_acc:.4f}")

# ── 6. Save best model ────────────────────────────────────────────────────────
model.load_state_dict(best_weights)
save_path = os.path.join(MODELS_OUT, "resnet50_finetuned.pth")
torch.save(model.state_dict(), save_path)
print(f"Model saved → {save_path}")

# ── 7. Training curves ────────────────────────────────────────────────────────
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

plt.suptitle("ResNet50 Fine-Tuning – Training Curves (Part 2)", fontsize=13)
plt.tight_layout()
curves_path = os.path.join(OUTPUT_DIR, "training_curves_part2.png")
plt.savefig(curves_path, dpi=150)
print(f"Training curves → {curves_path}")

# ── 8. Confusion matrix on validation set ────────────────────────────────────
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs.to(device))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True",      fontsize=12)
ax.set_title("Confusion Matrix – Fine-Tuned ResNet50 (Part 2)", fontsize=13)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_part2.png")
plt.savefig(cm_path, dpi=150)
print(f"Confusion matrix  → {cm_path}")

# ── 9. Classification report ──────────────────────────────────────────────────
report = classification_report(all_labels, all_preds,
                                target_names=class_names, zero_division=0)
print("\nClassification Report:\n")
print(report)

report_path = os.path.join(OUTPUT_DIR, "classification_report_part2.txt")
with open(report_path, "w") as fh:
    fh.write("ResNet50 Fine-Tuning – Classification Report\n")
    fh.write("=" * 60 + "\n\n")
    fh.write(f"Best validation accuracy: {best_acc:.4f}\n\n")
    fh.write(report)
print(f"Classification report → {report_path}")
