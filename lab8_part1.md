# Lab 8 – Part 1: ResNet50 Inference & Evaluation

## Objectives

- Load a pretrained **ResNet50** (trained on ImageNet-1K).
- Run inference on **10 sample images** from a 4-class dataset (cats / dogs / horses / Humans).
- Map the 1000-class ImageNet predictions to our 4 classes.
- Generate a **confusion matrix** and a **per-class classification report**.

---

## 1. Directory Structure

```
Lab8/try_2026/
├── data/
│   └── data/
│       ├── cats/          (202 images)
│       ├── dogs/          (202 images)
│       ├── horses/        (202 images)
│       └── Humans/        (202 images)
├── lab8_python/           ← created by install_offline.ps1 (Python 3.12, portable)
├── offline_packages/
│   ├── python_embedded/   ← python-3.12.9-embed-amd64.zip  (no install needed)
│   ├── pip_packages/      ← CPU-only .whl files (cp312/win_amd64)
│   └── models/
│       └── hub/
│           └── checkpoints/ ← resnet50-0676ba61.pth
├── output/                ← results written here (auto-created)
├── setup/
│   ├── download_resources.py   ← instructor runs on connected machine
│   ├── install_offline.ps1     ← students run once to set up lab8_python/
│   ├── activate.ps1            ← students dot-source before each session
│   └── install_offline.sh      ← Linux/macOS setup
├── inference.py           ← Part 1 student skeleton
├── finetune.py            ← Part 2 student skeleton
├── concept.md             ← required functions summary
└── solution/              ← reference implementations
```

---

## 2. Environment Setup

> **No Python installation or admin rights required.**
> The lab uses a self-contained Python 3.12 embeddable package that lives
> entirely inside the `lab8_python/` folder.

### Option A – Offline (lab machines, no internet)

> The instructor has prepared an `offline_packages/` folder on a USB drive
> or network share.  Copy it into the `try_2026/` folder before proceeding.

**Step 1 – Run setup once per machine**

Open PowerShell, navigate to `try_2026/`, then run:

```powershell
powershell -ExecutionPolicy Bypass -File setup\install_offline.ps1
```

This will:
- Extract `lab8_python\python.exe` (Python 3.12, no install needed)
- Bootstrap pip from the local wheel
- Install all packages (torch CPU, numpy, matplotlib, etc.) into `lab8_python\`

**Step 2 – Activate at the start of each session**

```powershell
. .\setup\activate.ps1
```

*(The leading dot `. ` sources the script into your current session.)*

You should see: `Lab 8 Python activated: Python 3.12.9`

**Step 3 – Verify the model weights are present**

```
offline_packages\models\hub\checkpoints\resnet50-0676ba61.pth  (~98 MB)
```

---

### Option B – Online (connected machine)

**Step 1 – Download and run setup**

```powershell
python setup/download_resources.py   # downloads everything to offline_packages/
powershell -ExecutionPolicy Bypass -File setup\install_offline.ps1
. .\setup\activate.ps1
```

---

## 3. Running Part 1 – Inference

After activating (`activate.ps1`), from the `try_2026/` directory complete the TODOs in `inference.py`, then run:

```powershell
python inference.py
```

### What the script does

| Step | Action |
|------|--------|
| 1 | Loads ResNet50 with ImageNet weights (from cache if offline) |
| 2 | Randomly selects **10 images** (2–3 per class, seeded for reproducibility) |
| 3 | Preprocesses each image (resize → crop → normalise) |
| 4 | Runs a forward pass; takes the **top-5 predictions** |
| 5 | Maps ImageNet class names → {cats, dogs, horses, Humans} using keyword matching |
| 6 | Computes confusion matrix and classification report |
| 7 | Saves all outputs to `output/` |

> A completed instructor/reference version is available in `solution/inference.py`.

### Expected console output

```
Device: cpu
Loading ResNet50 (ImageNet weights)…

Test set (10 images):
  [cats    ] cat.42.jpg
  [cats    ] cat.7.jpg
  [cats    ] cat.15.jpg
  ...

Running inference…

  TRUE=cats      PRED=cats      top1='tabby' (78.3%)
  TRUE=dogs      PRED=dogs      top1='golden_retriever' (65.1%)
  ...

Confusion matrix  → output/confusion_matrix_part1.png
Classification report → output/classification_report_part1.txt
```

---

## 4. Output Files

After running, check the `output/` directory:

| File | Description |
|------|-------------|
| `confusion_matrix_part1.png` | 4×4 heatmap of true vs predicted classes |
| `classification_report_part1.txt` | Precision / recall / F1 per class + per-image detail |

### Sample confusion matrix

```
             Predicted
           cats  dogs  horses  Humans
True cats  [  2     0       0       1 ]
True dogs  [  0     3       0       0 ]
True horses[  0     0       2       0 ]
True Humans[  0     0       0       2 ]
```

> ⚠️ **Note:** ResNet50 is not specifically trained on our 4 classes.
> "Humans" images are especially challenging because ImageNet contains
> few explicitly human-labelled classes.  Low accuracy for Humans is
> expected and is a motivation for Part 2 (fine-tuning).

---

## 5. Lab Report Requirements

Submit a PDF report containing the following sections:

### 5.1 Introduction (½ page)
- Brief description of ResNet50 architecture (depth, residual connections).
- Explain what "inference on a pretrained model" means.

### 5.2 Methodology
- List the 10 images you tested (class and filename).
- Describe the ImageNet→4-class mapping strategy used in `inference.py`.

### 5.3 Results
- Include the **confusion matrix image** (`confusion_matrix_part1.png`).
- Include the **classification report** (copy from the `.txt` file).
- Fill in the table below:

| Class  | Precision | Recall | F1-score | # Images tested |
|--------|-----------|--------|----------|-----------------|
| cats   |           |        |          |                 |
| dogs   |           |        |          |                 |
| horses |           |        |          |                 |
| Humans |           |        |          |                 |

### 5.4 Discussion (1 page)
Answer these questions:
1. Which class was predicted most accurately? Why?
2. Which class was predicted least accurately? Why?
3. What does a large number on the off-diagonal of the confusion matrix indicate?
4. Why does the pretrained model struggle with "Humans"? What would you do to improve it?

### 5.5 Conclusion (¼ page)
- Summarise the key finding: is a pretrained ImageNet model directly
  usable for this 4-class task?

---

## 6. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Activate the virtual environment |
| Weights not found / download error | Run in offline mode with weights in `offline_packages/models/hub/checkpoints/` |
| `FileNotFoundError: data/data/cats` | Make sure you are running from the `try_2026/` directory |
| CUDA out of memory | The script auto-falls back to CPU; no action needed |
| Slow inference | Normal on CPU (~1–2 s/image); GPU not required for Part 1 |
