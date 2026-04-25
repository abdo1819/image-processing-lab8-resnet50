"""
Lab 8 – Part 2 student skeleton.

Complete the TODOs in this file to build the fine-tuning workflow.
A complete reference implementation is available in solution/finetune.py.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
MODELS_OUT = os.path.join(SCRIPT_DIR, "trained_models")
OFFLINE_MODELS = os.path.join(SCRIPT_DIR, "offline_packages", "models")

os.environ["TORCH_HOME"] = OFFLINE_MODELS
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_OUT, exist_ok=True)

NUM_CLASSES = 4
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42


class SplitDataset(torch.utils.data.Dataset):
    """Wrap a subset of ImageFolder samples and apply a transform per split."""

    def __init__(self, folder: ImageFolder, indices: list[int], transform):
        self.folder = folder
        self.indices = indices
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        raise NotImplementedError("TODO: return the number of selected samples.")

    def __getitem__(self, idx):
        raise NotImplementedError("TODO: load one sample, apply the transform, and return image + label.")



def get_device() -> torch.device:
    """Return the device that should be used for training and evaluation."""
    raise NotImplementedError("TODO: select cuda when available, otherwise cpu.")



def build_transforms():
    """Create the training and validation transform pipelines."""
    raise NotImplementedError("TODO: define augmentation for train and normalization for both splits.")



def create_dataloaders(data_dir: str, train_tf, val_tf):
    """Build the dataset split and return train/validation loaders plus class names."""
    raise NotImplementedError("TODO: create ImageFolder, split indices deterministically, and build loaders.")



def build_model(device: torch.device):
    """Load ResNet50, freeze the backbone, and replace the classifier head."""
    raise NotImplementedError("TODO: load pretrained weights, freeze parameters, and replace model.fc.")



def train_model(model, train_loader, val_loader, device: torch.device):
    """Train the classifier head and return the best model state plus training history."""
    raise NotImplementedError("TODO: implement the epoch loop, optimizer, scheduler, and validation tracking.")



def save_model(model, output_path: str):
    """Save the trained weights to disk."""
    raise NotImplementedError("TODO: write the model state dict to trained_models/.")



def save_training_curves(history: dict, output_dir: str):
    """Plot and save the loss/accuracy curves."""
    raise NotImplementedError("TODO: generate the training curve figure.")



def evaluate_model(model, val_loader, device: torch.device):
    """Run the best model on the validation loader and collect predictions."""
    raise NotImplementedError("TODO: gather validation predictions and labels.")



def save_confusion_matrix(all_labels, all_preds, class_names, output_dir: str):
    """Create and save the validation confusion matrix figure."""
    raise NotImplementedError("TODO: generate the confusion matrix heatmap.")



def save_classification_report(all_labels, all_preds, class_names, best_acc: float, output_dir: str):
    """Write the validation classification report to disk."""
    raise NotImplementedError("TODO: save the text report with the best validation accuracy.")



def main() -> None:
    """Coordinate the full fine-tuning workflow."""
    raise NotImplementedError("TODO: connect the helper functions to complete Part 2.")


if __name__ == "__main__":
    main()
