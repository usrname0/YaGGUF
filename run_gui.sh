#!/bin/bash
# Quick launcher for the GUI on Unix/Mac

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "========================================"
    echo "Virtual Environment Not Found"
    echo "========================================"
    echo ""
    echo "This is your first time running the GUI."
    echo "We need to set up the environment first."
    echo ""
    printf "Press Enter to run setup now, or Ctrl+C to cancel..."
    read _
    echo ""
    bash setup/setup_linux.sh
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

echo "Starting GGUF Converter GUI..."
echo ""
echo "Opening browser to http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run gguf_converter/gui.py
