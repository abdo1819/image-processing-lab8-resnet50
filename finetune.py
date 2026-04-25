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

YOUR TASK
---------
  TODO – Complete sections 2, 3, 4, 5, and 6 marked below:
    2. Freeze the backbone and replace the FC head.
    3. Define the loss function and optimizer.
    4. Fill in the training/validation batch loop body.
    5. Save the trained model.
    6. Plot and save the training curves.
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

# ── 1. Dataset (provided – do not modify) ────────────────────────────────────
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

# ── 2. TODO: Model – freeze backbone, replace FC head ─────────────────────────
# a) Load ResNet50 with IMAGENET1K_V1 weights.
# b) Freeze ALL parameters (loop: set param.requires_grad = False).
# c) Replace model.fc with nn.Linear(model.fc.in_features, NUM_CLASSES).
# d) Move the model to `device`.
# e) Print how many parameters are trainable vs. total (see lab instructions).
#
#   Hint – after freezing, only model.fc has requires_grad=True because
#   you assigned a brand-new layer in step (c).

print("\nLoading ResNet50 (ImageNet weights)…")

# --- your code here ---
# model = ...

# ── 3. TODO: Loss function and optimizer ──────────────────────────────────────
# Use CrossEntropyLoss and Adam.
# The optimizer should update ONLY model.fc parameters.

# --- your code here ---
# criterion = ...
# optimizer = ...

# ── 4. Training loop (outer structure provided) ───────────────────────────────
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print()
for epoch in range(NUM_EPOCHS):
    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}]")

    for phase in ("train", "val"):
        model.train() if phase == "train" else model.eval()
        loader = train_loader if phase == "train" else val_loader

        total_loss, correct = 0.0, 0

        # TODO: Iterate over `loader`; for each batch (inputs, labels):
        #   • Move inputs and labels to `device`.
        #   • Zero the gradients.
        #   • Forward pass (use torch.set_grad_enabled for train vs val).
        #   • Compute the loss with `criterion`.
        #   • Get predictions with torch.max.
        #   • If training: back-propagate and step the optimizer.
        #   • Accumulate total_loss and correct counts.
        #
        # After the loop compute epoch_loss and epoch_acc, append to history,
        # and print:  "  train  loss=X.XXXX  acc=X.XXXX"

        # --- your code here ---

# ── 5. TODO: Save the trained model ──────────────────────────────────────────
# Save model.state_dict() to  trained_models/resnet50_finetuned.pth
# and print the save path.

# --- your code here ---

# ── 6. TODO: Plot training curves ────────────────────────────────────────────
# Create a 1×2 figure:
#   Left panel  – train loss vs val loss over epochs.
#   Right panel – train acc  vs val acc  over epochs.
# Save to  output/training_curves_part2.png  and print the save path.

# --- your code here ---
