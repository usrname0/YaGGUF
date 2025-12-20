#!/bin/bash
# Update dependencies and restart Streamlit

# Change to project root
cd "$(dirname "$0")/.."

echo ""
echo "========================================"
echo "Updating Dependencies and Restarting"
echo "========================================"
echo ""

# Wait for Streamlit to fully exit
sleep 2

# Activate venv
source venv/bin/activate

# Update all dependencies
echo "Updating dependencies from requirements.txt..."
python -m pip install --upgrade -r requirements.txt

echo ""
echo "========================================"
echo "Update complete! Restarting GUI..."
echo "========================================"
echo ""

# Restart Streamlit
streamlit run gguf_converter/gui.py
