"""
Lab 8 – Part 1: ResNet50 Inference on 4-Class Dataset
======================================================
Loads a pretrained ResNet50 (ImageNet weights), runs inference on 10
sample images drawn from the 4 lab classes (cats / dogs / horses / Humans),
and prints the top-5 ImageNet predictions for each image.

Offline mode: set TORCH_HOME to offline_packages/models so PyTorch
reads the cached weights instead of downloading them.
"""

import os
import random
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(SCRIPT_DIR, "data", "data")
OUTPUT_DIR       = os.path.join(SCRIPT_DIR, "output")
OFFLINE_MODELS   = os.path.join(SCRIPT_DIR, "offline_packages", "models")

# Point PyTorch to the offline cache (works online too – just a cache hint)
os.environ["TORCH_HOME"] = OFFLINE_MODELS
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
CLASSES    = ["cats", "dogs", "horses", "Humans"]  # must match folder names
NUM_IMAGES = 10    # total test images (split evenly across 4 classes)
SEED       = 42

# ── 1. Device ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 2. Load pretrained ResNet50 ───────────────────────────────────────────────
print("Loading ResNet50 (ImageNet weights)…")
weights = models.ResNet50_Weights.IMAGENET1K_V1
model   = models.resnet50(weights=weights)
model.eval()
model.to(device)

# Built-in ImageNet class names (no extra file needed)
imagenet_classes = weights.meta["categories"]   # list of 1000 strings

# ── 3. Preprocessing (ImageNet standard) ─────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── 4. Select 10 test images (balanced) ──────────────────────────────────────
random.seed(SEED)
test_images = []   # list of (abs_path, true_class_name)
per_class   = NUM_IMAGES // len(CLASSES)        # 2 per class
extras      = NUM_IMAGES  % len(CLASSES)        # 2 extra for first two classes

for i, cls in enumerate(CLASSES):
    cls_dir = os.path.join(DATA_DIR, cls)
    files   = sorted([
        f for f in os.listdir(cls_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    n       = per_class + (1 if i < extras else 0)
    chosen  = random.sample(files, min(n, len(files)))
    for f in chosen:
        test_images.append((os.path.join(cls_dir, f), cls))

print(f"\nTest set ({len(test_images)} images):")
for p, c in test_images:
    print(f"  [{c:8s}] {os.path.basename(p)}")

# ── 5. Run inference ──────────────────────────────────────────────────────────
print("\nRunning inference…\n")
for img_path, true_cls in test_images:
    img    = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        top5   = torch.topk(probs, 5)

    top5_names = [imagenet_classes[i.item()] for i in top5.indices]
    conf       = top5.values[0].item() * 100
    print(f"  TRUE={true_cls:8s}  top1='{top5_names[0]}' ({conf:.1f}%)")
    print(f"           top5: {top5_names}")
    print()
