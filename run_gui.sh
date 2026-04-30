#!/bin/bash
# Quick launcher for the GUI on Unix/Mac

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "========================================"
    echo "Installing YaGGUF"
    echo "========================================"
    echo ""
    echo "No existing installation was found. A fresh environment will be created."
    echo ""
    printf "Press Enter to continue, or Ctrl+C to cancel..."
    read _
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
python scripts/check_and_download_binaries.py
echo ""

echo "Starting YaGGUF GUI..."
echo ""
echo "Streamlit will open in your browser automatically."
echo "The URL will be displayed below."
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run gguf_converter/gui.py --server.address=localhost
