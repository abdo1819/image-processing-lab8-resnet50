"""
Lab 8 – Part 1: ResNet50 Inference on 4-Class Dataset
======================================================
Loads a pretrained ResNet50 (ImageNet weights), runs inference on 10
sample images drawn from the 4 lab classes (cats / dogs / horses / Humans),
maps ImageNet predictions back to those 4 classes, then produces:
  • output/confusion_matrix_part1.png
  • output/classification_report_part1.txt

Offline mode: set TORCH_HOME to offline_packages/models so PyTorch
reads the cached weights instead of downloading them.
"""

import os
import random
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(SCRIPT_DIR, "data", "data")
OUTPUT_DIR       = os.path.join(SCRIPT_DIR, "output")
OFFLINE_MODELS   = os.path.join(SCRIPT_DIR, "offline_packages", "models")

# Point PyTorch to the offline cache (works online too – just a cache hint)
os.environ["TORCH_HOME"] = OFFLINE_MODELS
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
CLASSES     = ["cats", "dogs", "horses", "Humans"]  # must match folder names
NUM_IMAGES  = 10    # total test images (split evenly across 4 classes)
SEED        = 42

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

# ── 5. ImageNet → 4-class mapping ────────────────────────────────────────────
#  We match keywords in the ImageNet class name to decide which of our 4
#  classes it belongs to.  Top-5 predictions are checked in order.

CAT_KW   = ["cat", "tabby", "kitten", "persian", "siamese", "egyptian_cat",
             "tiger_cat", "cougar", "lynx", "leopard", "cheetah"]
DOG_KW   = ["dog", "hound", "terrier", "spaniel", "retriever", "poodle",
             "husky", "bulldog", "beagle", "dachshund", "chihuahua", "pug",
             "setter", "shepherd", "mastiff", "samoyed", "collie", "boxer"]
HORSE_KW = ["horse", "mare", "stallion", "pony", "colt", "foal", "sorrel",
             "appaloosa", "arabian"]
HUMAN_KW = ["person", "man", "woman", "human", "player", "gymnast",
             "cowboy", "baseball", "basketball", "bride", "groom", "soldier"]

def map_imagenet_to_class(name: str) -> str:
    n = name.lower()
    if any(k in n for k in CAT_KW):   return "cats"
    if any(k in n for k in DOG_KW):   return "dogs"
    if any(k in n for k in HORSE_KW): return "horses"
    if any(k in n for k in HUMAN_KW): return "Humans"
    return "unknown"

# ── 6. Run inference ──────────────────────────────────────────────────────────
true_labels = []
pred_labels = []
detail_rows = []

print("\nRunning inference…\n")
for img_path, true_cls in test_images:
    img    = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        top5   = torch.topk(probs, 5)

    # Walk top-5 until a mapped class is found
    pred_cls   = "unknown"
    top1_name  = imagenet_classes[top5.indices[0].item()]
    for idx in top5.indices.tolist():
        mapped = map_imagenet_to_class(imagenet_classes[idx])
        if mapped != "unknown":
            pred_cls = mapped
            break

    # Fallback: assign the most probable class naively
    if pred_cls == "unknown":
        pred_cls = "cats"   # default; edit if desired

    true_labels.append(true_cls)
    pred_labels.append(pred_cls)

    conf = top5.values[0].item() * 100
    print(f"  TRUE={true_cls:8s}  PRED={pred_cls:8s}  "
          f"top1='{top1_name}' ({conf:.1f}%)")
    detail_rows.append({
        "image":         os.path.basename(img_path),
        "true":          true_cls,
        "pred":          pred_cls,
        "top1_imagenet": top1_name,
        "confidence":    f"{conf:.1f}%",
    })

# ── 7. Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(true_labels, pred_labels, labels=CLASSES)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True",      fontsize=12)
ax.set_title("Confusion Matrix – ResNet50 Inference (Part 1)", fontsize=13)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_part1.png")
plt.savefig(cm_path, dpi=150)
print(f"\nConfusion matrix  → {cm_path}")

# ── 8. Classification report ──────────────────────────────────────────────────
report = classification_report(true_labels, pred_labels,
                                labels=CLASSES, zero_division=0)
print("\nClassification Report:\n")
print(report)

report_path = os.path.join(OUTPUT_DIR, "classification_report_part1.txt")
with open(report_path, "w") as fh:
    fh.write("ResNet50 Inference – Classification Report\n")
    fh.write("=" * 60 + "\n\n")
    fh.write(report)
    fh.write("\n\nPer-image results:\n")
    fh.write(f"{'Image':<35} {'True':>8} {'Pred':>8}  Top-1 ImageNet\n")
    fh.write("-" * 80 + "\n")
    for r in detail_rows:
        fh.write(f"{r['image']:<35} {r['true']:>8} {r['pred']:>8}  "
                 f"{r['top1_imagenet']}  ({r['confidence']})\n")
print(f"Classification report → {report_path}")
