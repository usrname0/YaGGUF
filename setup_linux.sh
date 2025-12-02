#!/bin/bash
echo "========================================"
echo "Yet Another GGUF Converter - Setup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Python found:"
python3 --version
echo ""

# Check Python version
python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if [ $? -ne 0 ]; then
    echo "ERROR: Python 3.8 or higher is required"
    exit 1
fi

echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing old venv..."
    rm -rf venv
fi

python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

echo ""
echo "Installing PyTorch (CPU-only version to save 1.8GB)..."
pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install PyTorch"
    exit 1
fi

echo ""
echo "Installing remaining requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements"
    exit 1
fi

echo ""
echo "Downloading llama.cpp binaries..."
python -m gguf_converter.download_binaries
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download binaries"
    echo "You can try running the setup again or install llama.cpp manually"
    exit 1
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To use the converter:"
echo "  - GUI: Run ./run_gui.sh"
echo "  - CLI: Run 'source venv/bin/activate', then use: python -m gguf_converter"
echo ""
