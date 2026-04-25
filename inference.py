"""
Lab 8 – Part 1 student skeleton.

Complete the TODOs in this file to build the inference pipeline.
A complete reference implementation is available in solution/inference.py.
"""

import os
import random
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
OFFLINE_MODELS = os.path.join(SCRIPT_DIR, "offline_packages", "models")

os.environ["TORCH_HOME"] = OFFLINE_MODELS
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ["cats", "dogs", "horses", "Humans"]
NUM_IMAGES = 10
SEED = 42

CAT_KW = ["cat", "tabby", "kitten", "persian", "siamese", "egyptian_cat",
          "tiger_cat", "cougar", "lynx", "leopard", "cheetah"]
DOG_KW = ["dog", "hound", "terrier", "spaniel", "retriever", "poodle",
          "husky", "bulldog", "beagle", "dachshund", "chihuahua", "pug",
          "setter", "shepherd", "mastiff", "samoyed", "collie", "boxer"]
HORSE_KW = ["horse", "mare", "stallion", "pony", "colt", "foal", "sorrel",
            "appaloosa", "arabian"]
HUMAN_KW = ["person", "man", "woman", "human", "player", "gymnast",
            "cowboy", "baseball", "basketball", "bride", "groom", "soldier"]


def get_device() -> torch.device:
    """Return the device that should be used for inference."""
    raise NotImplementedError("TODO: select cuda when available, otherwise cpu.")



def load_pretrained_model(device: torch.device):
    """Load ResNet50 ImageNet weights and move the model to the target device."""
    raise NotImplementedError("TODO: load ResNet50 pretrained weights and return model, weights.")



def build_preprocess():
    """Create the ImageNet preprocessing pipeline for evaluation."""
    raise NotImplementedError("TODO: define resize, crop, tensor conversion, and normalization.")



def select_test_images(data_dir: str, classes: list[str], num_images: int, seed: int):
    """Pick a balanced set of image paths and their labels from the dataset."""
    raise NotImplementedError("TODO: sample images from each class folder.")



def map_imagenet_to_class(name: str) -> str:
    """Map one ImageNet class label to one of the four lab classes."""
    raise NotImplementedError("TODO: use the keyword lists above to return a lab class or 'unknown'.")



def run_inference(model, preprocess, imagenet_classes, device, test_images):
    """Run inference on the selected images and collect prediction details."""
    raise NotImplementedError("TODO: execute forward passes, inspect top-5 predictions, and collect rows.")



def save_confusion_matrix(true_labels: list[str], pred_labels: list[str], output_dir: str):
    """Create and save the confusion matrix figure."""
    raise NotImplementedError("TODO: generate a confusion matrix heatmap and save it in output/.")



def save_classification_report(true_labels: list[str], pred_labels: list[str], detail_rows: list[dict], output_dir: str):
    """Write the classification report and per-image details to disk."""
    raise NotImplementedError("TODO: create the report text file in output/.")



def main() -> None:
    """Coordinate the full inference workflow."""
    random.seed(SEED)
    raise NotImplementedError("TODO: call the helper functions in order to complete Part 1.")


if __name__ == "__main__":
    main()
