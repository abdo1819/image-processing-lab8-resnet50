"""
Lab 8 – Offline Resource Downloader
=====================================
Run this script ONCE on any machine WITH internet access.
It downloads everything needed for students to run the lab offline:

  1. Python 3.12 embeddable ZIP  → offline_packages/python_embedded/
       No installation or admin rights needed on student machines.
       Students just run install_offline.ps1 which extracts and configures it.

  2. pip + all required wheels   → offline_packages/pip_packages/
       Pinned to cp312 / win_amd64 to match the embeddable Python exactly.

  3. ResNet50 pretrained weights → offline_packages/models/hub/checkpoints/

Copy the entire offline_packages/ folder to a USB drive or network share,
then run  setup/install_offline.ps1  on each student machine.

Usage:
    python setup/download_resources.py
"""

import os
import sys
import subprocess
import urllib.request

# ── Configuration ─────────────────────────────────────────────────────────────
PYTHON_VERSION   = "3.12.9"
PYTHON_EMBED_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/"
    f"python-{PYTHON_VERSION}-embed-amd64.zip"
)

RESNET50_URL      = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
RESNET50_FILENAME = "resnet50-0676ba61.pth"

CPU_INDEX    = "https://download.pytorch.org/whl/cpu"
PY_VER_SHORT = "3.12"
PLATFORM     = "win_amd64"

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_DIR = os.path.join(ROOT_DIR, "offline_packages", "python_embedded")
PIP_DIR   = os.path.join(ROOT_DIR, "offline_packages", "pip_packages")
MODEL_DIR = os.path.join(ROOT_DIR, "offline_packages", "models")

for d in (EMBED_DIR, PIP_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar  = "#" * int(pct / 2)
        print(f"\r  [{bar:<50}] {pct:5.1f}%", end="", flush=True)


def download(url, dest, label):
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        print(f"  Already exists, skipping: {os.path.basename(dest)}")
        return
    print(f"  {label}")
    print(f"  {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()
    print(f"  -> {os.path.getsize(dest)/1e6:.1f} MB saved to {dest}")


# ── Step 1: Python 3.12 embeddable ZIP ───────────────────────────────────────
print("=" * 60)
print(f" Step 1 – Python {PYTHON_VERSION} embeddable ZIP (~12 MB)")
print("=" * 60)
download(
    PYTHON_EMBED_URL,
    os.path.join(EMBED_DIR, f"python-{PYTHON_VERSION}-embed-amd64.zip"),
    f"Python {PYTHON_VERSION} embeddable package",
)
print()

# ── Step 2: pip wheels pinned to cp312 / win_amd64 ───────────────────────────
# pip itself is included so we can bootstrap it into the embeddable Python.
TORCH_PKGS = ["torch", "torchvision"]
OTHER_PKGS = ["pip", "setuptools", "wheel",
              "numpy", "matplotlib", "seaborn",
              "scikit-learn", "Pillow", "tqdm"]

BASE_FLAGS = [
    sys.executable, "-m", "pip", "download",
    "--python-version", PY_VER_SHORT,
    "--platform",       PLATFORM,
    "--only-binary",    ":all:",
    "--dest",           PIP_DIR,
]

print("=" * 60)
print(f" Step 2a – CPU-only torch + torchvision  (cp312/{PLATFORM})")
print(f"           {CPU_INDEX}")
print("=" * 60)
subprocess.check_call([*BASE_FLAGS, "--index-url", CPU_INDEX, *TORCH_PKGS])
print()

print("=" * 60)
print(f" Step 2b – Remaining packages  (cp312/{PLATFORM})")
print("=" * 60)
subprocess.check_call([*BASE_FLAGS, *OTHER_PKGS])
print(f"\n  -> All wheels in: {PIP_DIR}\n")

# ── Step 3: ResNet50 weights ──────────────────────────────────────────────────
print("=" * 60)
print(" Step 3 – ResNet50 ImageNet weights  (~98 MB)")
print("=" * 60)
checkpoints = os.path.join(MODEL_DIR, "hub", "checkpoints")
os.makedirs(checkpoints, exist_ok=True)
download(RESNET50_URL, os.path.join(checkpoints, RESNET50_FILENAME), "ResNet50 weights")
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print(" ALL DONE")
print("=" * 60)
print(f"""
Copy this folder to a USB drive or network share:
  {os.path.join(ROOT_DIR, 'offline_packages')}

Folder structure:
  offline_packages/
    python_embedded/
      python-{PYTHON_VERSION}-embed-amd64.zip  <- self-contained Python (no install)
    pip_packages/
      pip-*.whl                                <- pip bootstrap wheel
      torch-*+cpu-cp312-*.whl                 <- CPU-only torch
      torchvision-*+cpu-cp312-*.whl
      ... (other wheels, all cp312/win_amd64)
    models/
      hub/checkpoints/
        {RESNET50_FILENAME}           <- ResNet50 weights

On each student machine run:
  powershell -ExecutionPolicy Bypass -File setup\\install_offline.ps1
""")

