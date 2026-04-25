# Lab 8 – Part 1: ResNet50 Inference & Evaluation

## Objectives

- Load a pretrained **ResNet50** (trained on ImageNet-1K).
- Run inference on **10 sample images** from a 4-class dataset (cats / dogs / horses / Humans).
- Observe the raw **ImageNet top-5 predictions** for each image.
- Manually record results and build a confusion matrix.

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
├── inference.py           ← Part 1 script
└── finetune.py            ← Part 2 script
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

After activating (`activate.ps1`), from the `try_2026/` directory run:

```powershell
python inference.py
```

### What the script does

| Step | Action |
|------|--------|
| 1 | Loads ResNet50 with ImageNet weights (from cache if offline) |
| 2 | Randomly selects **10 images** (2–3 per class, seeded for reproducibility) |
| 3 | Preprocesses each image (resize → crop → normalise) |
| 4 | Runs a forward pass; prints the **top-5 ImageNet predictions** and confidence |

### Expected console output

```
Device: cpu
Loading ResNet50 (ImageNet weights)…

Test set (10 images):
  [cats    ] cat.42.jpg
  [cats    ] cat.7.jpg
  ...

Running inference…

  TRUE=cats      top1='tabby' (78.3%)
           top5: ['tabby', 'tiger_cat', 'Egyptian_cat', 'lynx', 'Persian_cat']

  TRUE=dogs      top1='golden_retriever' (65.1%)
           top5: ['golden_retriever', 'Labrador_retriever', 'cocker_spaniel', ...]
  ...
```

---

## 4. Output Files

The script produces no output files — all results are printed to the console.
Record the printed predictions manually (see Section 5 below).

---

## 5. Building the Confusion Matrix Manually

After running `inference.py`, the console shows each image's true class and the
top-5 ImageNet predictions.

**Step 1 – Decide the predicted class.**  
Look at the top-5 names and pick the entry that best matches one of:
`cats`, `dogs`, `horses`, `Humans`.

> **Mapping hint (not in the script):**  
> The original script used keyword lists to do this automatically:  
> - **cats** → ImageNet names containing: `cat`, `tabby`, `kitten`, `persian`, `siamese`, `lynx`, `leopard`, …  
> - **dogs** → `dog`, `hound`, `terrier`, `spaniel`, `retriever`, `poodle`, `husky`, …  
> - **horses** → `horse`, `mare`, `pony`, `colt`, `foal`, `sorrel`, `appaloosa`, …  
> - **Humans** → `person`, `man`, `woman`, `player`, `gymnast`, `cowboy`, `soldier`, …  
> Use this as a guide when reading the raw output.

**Step 2 – Fill in the table.**

| Image | True class | Top-1 ImageNet | Predicted class |
|-------|-----------|----------------|-----------------|
| …     | cats      | tabby          | cats            |
| …     | Humans    | suit           | ?               |

**Step 3 – Build the 4×4 confusion matrix.**

Draw a grid with rows = True class, columns = Predicted class, and tally counts:

```
              Predicted
           cats  dogs  horses  Humans
True cats  [  ?     ?       ?       ? ]
True dogs  [  ?     ?       ?       ? ]
True horses[  ?     ?       ?       ? ]
True Humans[  ?     ?       ?       ? ]
```

**Step 4 – Compute per-class metrics.**

For each class `C`:

| Metric    | Formula |
|-----------|---------|
| Precision | TP / (TP + FP) |
| Recall    | TP / (TP + FN) |
| F1-score  | 2 × Precision × Recall / (Precision + Recall) |

where TP = diagonal cell for C, FP = column sum − TP, FN = row sum − TP.

---

## 5. Lab Report Requirements

Submit a PDF report containing the following sections:

### 5.1 Introduction (½ page)
- Brief description of ResNet50 architecture (depth, residual connections).
- Explain what "inference on a pretrained model" means.

### 5.2 Methodology
- List the 10 images you tested (class and filename, copied from the console output).
- Describe the ImageNet→4-class mapping strategy you used (keyword-based or your own).

### 5.3 Results
- Include the **manually drawn confusion matrix** (see Section 5 above).
- Fill in the table below (computed by hand from the confusion matrix):

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
