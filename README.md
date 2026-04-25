# Image Processing Lab 8 – ResNet50

This repository contains a two-part lab built around **ResNet50** for a 4-class image classification task:

- **Part 1:** run inference with a pretrained ImageNet ResNet50 model
- **Part 2:** fine-tune only the final classifier layer for the lab dataset

The target classes are:

- `cats`
- `dogs`
- `horses`
- `Humans`

> `Humans` is intentionally capitalized because the scripts expect that exact
> folder/class name.

## Repository contents

```text
.
├── data/                  # Dataset root
├── inference.py           # Part 1 student skeleton
├── finetune.py            # Part 2 student skeleton
├── concept.md             # Required functions and their roles
├── output/                # Generated reports/plots
├── solution/              # Reference implementations
├── trained_models/        # Saved fine-tuned weights
├── setup/                 # Setup and offline resource scripts
├── lab8_part1.md          # Detailed instructions for Part 1
└── lab8_part2.md          # Detailed instructions for Part 2
```

## Expected dataset layout

Both scripts expect the dataset under:

```text
data/data/
├── cats/
├── dogs/
├── horses/
└── Humans/
```

## Environment setup

### Windows offline setup

Run once:

```powershell
powershell -ExecutionPolicy Bypass -File setup\install_offline.ps1
```

Activate for each session:

```powershell
. .\setup\activate.ps1
```

### Linux/macOS offline setup

```bash
chmod +x setup/install_offline.sh
./setup/install_offline.sh
```

### Download offline resources on a connected machine

```bash
python setup/download_resources.py
```

This prepares offline Python/wheels/model resources inside `offline_packages/`.

## Student workflow

- Students complete the TODOs in the root `inference.py` and `finetune.py` files.
- `concept.md` explains the required functions in both skeleton files.
- Instructors can use `solution/inference.py` and `solution/finetune.py` as the completed reference versions.

## Run Part 1: inference

```bash
python inference.py
```

What it does:

- loads pretrained **ResNet50** ImageNet weights
- samples 10 images from the dataset
- maps ImageNet predictions to the 4 lab classes
- saves:
  - `output/confusion_matrix_part1.png`
  - `output/classification_report_part1.txt`

Reference solution:

```bash
python solution/inference.py
```

## Run Part 2: fine-tuning

```bash
python finetune.py
```

What it does:

- loads pretrained **ResNet50**
- freezes the backbone
- replaces `model.fc` with a 4-class classifier
- trains only the new classifier head
- saves:
  - `output/training_curves_part2.png`
  - `output/confusion_matrix_part2.png`
  - `output/classification_report_part2.txt`
  - `trained_models/resnet50_finetuned.pth`

Reference solution:

```bash
python solution/finetune.py
```

## Notes

- Both scripts automatically use `cuda` if available, otherwise `cpu`.
- PyTorch is pointed at `offline_packages/models` through `TORCH_HOME`.
- Detailed lab instructions are available in:
  - `lab8_part1.md`
  - `lab8_part2.md`
