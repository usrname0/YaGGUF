#!/bin/bash
# Quick launcher for the GUI on Unix/Mac

GPU_BACKEND=""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "========================================"
    echo "Installing YaGGUF"
    echo "========================================"
    echo ""
    echo "No existing installation was found. A fresh environment will be created."
    echo "Press Ctrl+C to cancel."
    echo ""

    if [ "$(uname)" = "Darwin" ]; then
        echo "macOS detected: Metal acceleration is built-in (no selection needed)."
        GPU_BACKEND="cpu"
    else
        echo "Select GPU backend for llama.cpp binaries:"
        echo "  1. CPU only (recommended)"
        echo "  2. CUDA 12.4 (NVIDIA)"
        echo "  3. CUDA 13.1 (NVIDIA)"
        echo "  4. Vulkan (NVIDIA/AMD/Intel)"
        echo "  5. HIP / ROCm (AMD)"
        echo ""
        read -p "Enter number [1]: " GPU_BACKEND_CHOICE
        GPU_BACKEND_CHOICE=${GPU_BACKEND_CHOICE:-1}
        case "$GPU_BACKEND_CHOICE" in
            1) GPU_BACKEND="cpu" ;;
            2) GPU_BACKEND="cuda-12.4" ;;
            3) GPU_BACKEND="cuda-13.1" ;;
            4) GPU_BACKEND="vulkan" ;;
            5) GPU_BACKEND="hip-radeon" ;;
            *) GPU_BACKEND="cpu" ;;
        esac
    fi
    echo ""

    bash scripts/setup_linux.sh
    if [ $? -ne 0 ]; then
        echo ""
        echo "Setup failed. Please check the errors above."
        exit 1
    fi
    echo ""
    echo "Setup complete! Starting GUI..."
    echo ""
fi

# Activate virtual environment
. venv/bin/activate

# Check and update binaries if needed
echo ""
if [ -n "$GPU_BACKEND" ]; then
    python scripts/check_and_download_binaries.py --backend "$GPU_BACKEND"
else
    python scripts/check_and_download_binaries.py
fi
echo ""

echo "Starting YaGGUF GUI..."
echo ""
echo "Streamlit will open in your browser automatically."
echo "The URL will be displayed below."
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run gguf_converter/gui.py --server.address=localhost
