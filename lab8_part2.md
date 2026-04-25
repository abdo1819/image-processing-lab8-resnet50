# Lab 8 – Part 2: ResNet50 Fine-Tuning (Classifier-Only)

## Objectives

- Understand **transfer learning** and **feature extraction**.
- Load a pretrained ResNet50 and **freeze all layers except the final classifier**.
- Train the new head on the 4-class dataset (cats / dogs / horses / Humans).
- Evaluate the fine-tuned model and compare it with the Part 1 baseline.

---

## 1. Key Concept – What Is Being Trained?

ResNet50 ends with a **fully-connected (FC) layer** that maps 2048 features → 1000 ImageNet classes.

In this part we:
1. **Freeze** every convolutional and batch-norm layer (weights are locked).
2. **Replace** the final FC layer: 2048 → **4 classes**.
3. **Train only** the new FC layer while the frozen backbone extracts features.

```
Input image
   │
   ▼
┌────────────────────────────────┐
│  ResNet50 backbone             │  ← FROZEN (pretrained ImageNet weights)
│  (conv1 → layer1 → ... → avgpool) │
└────────────────────────────────┘
   │  2048-dim feature vector
   ▼
┌────────────────────────────────┐
│  Linear(2048 → 4)              │  ← TRAINABLE  (only this layer updates)
└────────────────────────────────┘
   │
   ▼
 cats / dogs / horses / Humans
```

This approach is extremely efficient: **~8 200 parameters trained** out of
~25 million total.

---

## 2. Environment Setup

The environment is shared with Part 1.  If you already ran
`install_offline.ps1` on this machine, just activate:

```powershell
. .\setup\activate.ps1
```

> **First time on this machine?**  Run the full setup first:
> ```powershell
> powershell -ExecutionPolicy Bypass -File setup\install_offline.ps1
> . .\setup\activate.ps1
> ```

### Offline model weights

The weights are the same ResNet50 file used in Part 1.  No additional
download is required.

```
offline_packages\models\hub\checkpoints\resnet50-0676ba61.pth   <- already present
```

---

## 3. Dataset Split

`finetune.py` automatically splits the data:

| Split | Fraction | Images per class | Total |
|-------|----------|-----------------|-------|
| Train | 80 %     | ~162            | ~648  |
| Val   | 20 %     | ~40             | ~160  |

The split is **deterministic** (fixed random seed = 42) so results are
reproducible across machines.

---

## 4. Hyperparameters

The default values are set at the top of `finetune.py`:

| Parameter    | Default | Explanation |
|--------------|---------|-------------|
| `BATCH_SIZE` | 16      | Images per gradient step |
| `NUM_EPOCHS` | 10      | Training passes over the dataset |
| `LR`         | 1e-3    | Adam learning rate |
| `VAL_SPLIT`  | 0.2     | Fraction held out for validation |
| `SEED`       | 42      | For reproducibility |

You can edit these values to experiment.

---

## 5. Running Part 2 – Fine-Tuning

With your virtual environment active, from the `try_2026/` directory run:

```bash
python finetune.py
```

> ⏱️ **Expected runtime:** ~2–5 minutes on CPU | ~30 seconds on GPU.

### What the script does

| Step | Action |
|------|--------|
| 1 | Loads ResNet50 backbone with frozen weights |
| 2 | Replaces `model.fc` with `Linear(2048 → 4)` |
| 3 | Applies data augmentation on the training split |
| 4 | Trains for 10 epochs; saves the best model by validation accuracy |
| 5 | Plots loss and accuracy curves |
| 6 | Evaluates the best model on the validation set |
| 7 | Saves confusion matrix and classification report |

### Expected console output (abbreviated)

```
Device: cpu
Classes: ['Humans', 'cats', 'dogs', 'horses']  |  Total images: 808
Train: 646 images  |  Val: 162 images

Loading ResNet50 (ImageNet weights)…
Trainable params: 8,196 / 25,557,032  (0.03%)

Epoch [01/10]  lr=1.0e-03
  train  loss=1.1234  acc=0.5232
  val    loss=0.9012  acc=0.6543
...
Epoch [10/10]  lr=1.0e-04
  train  loss=0.3210  acc=0.9010
  val    loss=0.3580  acc=0.8950

Best validation accuracy: 0.8950
Model saved → trained_models/resnet50_finetuned.pth
```

---

## 6. Output Files

| File | Description |
|------|-------------|
| `output/training_curves_part2.png` | Loss and accuracy curves (train vs val) |
| `output/confusion_matrix_part2.png` | 4×4 confusion matrix on the validation set |
| `output/classification_report_part2.txt` | Precision / recall / F1 per class |
| `trained_models/resnet50_finetuned.pth` | Saved model weights |

---

## 7. Lab Report Requirements

Submit a **combined** report for Parts 1 and 2, or a standalone Part 2
report – follow your instructor's instructions.

### 7.1 Introduction (½ page)
- Explain **transfer learning** and **feature extraction** vs full fine-tuning.
- Why do we freeze the backbone?

### 7.2 Methodology
- Describe the training setup (optimizer, loss function, scheduler, augmentation).
- Explain the train/val split strategy.

### 7.3 Results

**Training curves** – include `training_curves_part2.png` and describe:
- Does the training loss decrease consistently?
- Is there a gap between training and validation accuracy (overfitting)?

**Confusion matrix** – include `confusion_matrix_part2.png`.

**Classification report** – copy the metrics into the table:

| Class  | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| cats   |           |        |          |         |
| dogs   |           |        |          |         |
| horses |           |        |          |         |
| Humans |           |        |          |         |

### 7.4 Comparison with Part 1

Fill in this comparison table:

| Metric | Part 1 (Pretrained, no tuning) | Part 2 (Fine-tuned head) |
|--------|-------------------------------|--------------------------|
| Overall accuracy | | |
| cats F1 | | |
| dogs F1 | | |
| horses F1 | | |
| Humans F1 | | |

### 7.5 Discussion (1 page)

Answer these questions:
1. How much did fine-tuning improve accuracy compared to Part 1?
2. Only **~8 200 parameters** were trained out of 25 million.  How is
   this possible yet still effective?
3. The training used **data augmentation** (random flip, rotation, colour
   jitter).  What is the purpose of each augmentation?
4. What would happen if you **unfroze the entire model** and trained all
   layers?  State expected advantages and risks.
5. The learning rate is reduced at epoch 5 (`StepLR, gamma=0.1`).  Why
   does reducing LR during training help?

### 7.6 Conclusion (¼ page)

- Does classifier-only fine-tuning achieve satisfactory accuracy on this
  4-class dataset?
- What is your recommendation for a production deployment?

---

## 8. Optional Experiments (Bonus)

Try the following modifications and report the change in accuracy:

| Experiment | Change in `finetune.py` | Expected effect |
|------------|------------------------|-----------------|
| More epochs | `NUM_EPOCHS = 20` | Higher accuracy (risk of overfitting) |
| Larger batch | `BATCH_SIZE = 32` | Faster training; may need more RAM |
| Unfreeze last block | Add `for p in model.layer4.parameters(): p.requires_grad = True` | Better feature adaptation |
| Lower LR | `LR = 1e-4` | Slower but potentially more stable |

---

## 9. Troubleshooting

| Problem | Solution |
|---------|----------|
| `RuntimeError: CUDA out of memory` | Reduce `BATCH_SIZE` to 8 |
| Very low accuracy after 10 epochs | Check that `DATA_DIR` points to a folder with 4 sub-folders; run from `try_2026/` |
| Training is very slow on CPU | Reduce `NUM_EPOCHS` to 5 for a quick test; or use GPU if available |
| Weights not found | Ensure `offline_packages/models/hub/checkpoints/resnet50-*.pth` exists |
| `num_workers` warning on Windows | Change `num_workers=2` to `num_workers=0` in `finetune.py` |
