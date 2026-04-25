#!/usr/bin/env bash
# Lab 8 – Offline Package Installer (Linux / macOS)
# ==================================================
# Run on the OFFLINE machine after copying offline_packages/.
#
# Usage:
#   chmod +x setup/install_offline.sh
#   ./setup/install_offline.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PIP_DIR="$ROOT_DIR/offline_packages/pip_packages"

echo "============================================"
echo " Lab 8 | Offline Setup"
echo "============================================"
echo

# ── Step 1: Check Python ──────────────────────────────────────────────────────
echo "[Step 1] Checking for Python 3..."
if ! command -v python3 &>/dev/null; then
    echo "  ERROR: python3 not found."
    echo "  On Ubuntu/Debian:  sudo apt-get install python3 python3-venv python3-pip"
    echo "  On Fedora/RHEL:    sudo dnf install python3"
    echo "  On macOS:          install from https://www.python.org or use Homebrew"
    exit 1
fi
python3 --version
echo "  Python found."
echo

# ── Step 2: Install pip packages ─────────────────────────────────────────────
echo "[Step 2] Installing CPU-only packages from local .whl files..."
if [ ! -d "$PIP_DIR" ]; then
    echo "ERROR: Folder not found: $PIP_DIR"
    echo "Make sure you copied offline_packages/ from the prepared media."
    exit 1
fi

echo "Source: $PIP_DIR"
echo

pip3 install --no-index --find-links="$PIP_DIR" \
    torch torchvision numpy matplotlib seaborn scikit-learn Pillow tqdm

echo
echo "============================================"
echo " Installation successful!"
echo " You can now run:"
echo "   python3 inference.py   (Part 1)"
echo "   python3 finetune.py    (Part 2)"
echo "============================================"
