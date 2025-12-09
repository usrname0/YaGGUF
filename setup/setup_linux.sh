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

# Check if virtual environment already exists and is valid
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "Virtual environment already exists, checking if valid..."
    . venv/bin/activate

    # Quick check if streamlit is installed (main dependency)
    if python -c "import streamlit" 2>/dev/null; then
        echo "Virtual environment is valid, skipping package installation"
        SKIP_VENV_SETUP=1
    else
        echo "Virtual environment is incomplete, will reinstall packages"
        SKIP_VENV_SETUP=0
    fi
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to create virtual environment"
        echo ""
        # Clean up partial venv directory so next run will retry
        if [ -d "venv" ]; then
            rm -rf venv
        fi
        echo "On Debian/Ubuntu systems, you may need to install python3-venv:"
        echo "  sudo apt install python3-venv"
        echo ""
        echo "On Fedora/RHEL systems:"
        echo "  sudo dnf install python3-virtualenv"
        echo ""
        echo "After installing, run this setup script again."
        exit 1
    fi

    echo ""
    echo "Activating virtual environment..."
    . venv/bin/activate
    SKIP_VENV_SETUP=0
fi

if [ "$SKIP_VENV_SETUP" = "0" ]; then
    echo ""
    echo "Upgrading pip..."
    python -m pip install --upgrade pip

    echo ""
    echo "Installing PyTorch (CPU-only version to save space)..."
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
fi

echo ""
echo "Checking llama.cpp binaries..."

# Check if binaries already exist
if python -c "from gguf_converter.binary_manager import BinaryManager; import sys; sys.exit(0 if BinaryManager()._binaries_exist() else 1)" 2>/dev/null; then
    echo "Binaries already installed, skipping download"
else
    echo "Downloading llama.cpp binaries..."
    python -m gguf_converter.download_binaries
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download binaries"
        echo "You can try running the setup again or install llama.cpp manually"
        exit 1
    fi
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
