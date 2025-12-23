#!/bin/bash
# Minimal setup script for Linux/macOS
# This script is called by run_gui.sh and should not be run directly.

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
pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu
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