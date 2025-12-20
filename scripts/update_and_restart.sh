#!/bin/bash
# Update dependencies and restart Streamlit

# Change to project root
cd "$(dirname "$0")/.."

echo ""
echo "========================================"
echo "Updating Dependencies and Restarting"
echo "========================================"
echo ""

# Wait for port 8501 to be released
echo "Waiting for Streamlit to shut down..."
while lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1 ; do
    sleep 1
done
echo "Port 8501 is free"

# Activate venv
source venv/bin/activate

# Update dependencies from requirements.txt
echo "Updating dependencies from requirements.txt..."
python -m pip install --upgrade -r requirements.txt

echo ""
echo "========================================"
echo "Update complete! Restarting GUI..."
echo "========================================"
echo ""

# Restart Streamlit without opening new browser tab
streamlit run gguf_converter/gui.py --server.headless=true
