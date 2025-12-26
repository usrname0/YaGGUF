#!/bin/bash
# Minimal setup script for Linux/macOS
# This script is called by run_gui.sh and should not be run directly.

# Ensure all shell scripts have executable permissions
# This handles cases where executable bit was lost (ZIP download, git config, etc.)
chmod +x run_gui.sh 2>/dev/null || true
chmod +x scripts/*.sh 2>/dev/null || true

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if [ $? -ne 0 ]; then
    echo "ERROR: Python 3.8 or higher is required."
    exit 1
fi

# Check if tkinter is available (required for folder browser)
echo "Checking for tkinter..."
python3 -c "import tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: tkinter is not installed. The folder browser will not work."
    echo "To install tkinter:"
    echo "  - Ubuntu/Debian: sudo apt install python3-tk"
    echo "  - Fedora/RHEL:   sudo dnf install python3-tkinter"
    echo "  - Arch:          sudo pacman -S tk"
    echo ""
    echo "You can still use the GUI by typing paths manually."
    echo ""
    read -p "Press Enter to continue setup anyway..."
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment."
    echo "On Debian/Ubuntu, you may need to install python3-venv: sudo apt install python3-venv"
    echo "On Fedora/RHEL, you may need to install python3-virtualenv: sudo dnf install python3-virtualenv"
    # Clean up partial venv directory so next run will retry
    if [ -d "venv" ]; then
        rm -rf venv
    fi
    exit 1
fi

# Activate virtual environment so pip installs into it
. venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing PyTorch..."
pip install "torch>=2.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0" --index-url https://download.pytorch.org/whl/cpu
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install PyTorch."
    exit 1
fi

echo "Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements."
    exit 1
fi

exit 0