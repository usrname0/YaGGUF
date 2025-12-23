#!/bin/bash
# Update dependencies and restart Streamlit

# Change to project root
cd "$(dirname "$0")/.."

# Get port from argument or default to 8501
PORT=${1:-8501}

echo ""
echo "========================================"
echo "Updating Dependencies and Restarting"
echo "========================================"
echo ""

# Kill any process using the specified port
echo "Checking for port $PORT..."
PID=$(lsof -ti:$PORT)
if [ ! -z "$PID" ]; then
    echo "Killing process using port $PORT (PID: $PID)"
    kill -9 $PID 2>/dev/null
    sleep 2
fi

# Verify port is free
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "WARNING: Port $PORT still in use. Proceeding anyway..."
else
    echo "Port $PORT is now free"
fi

# Activate venv
source venv/bin/activate

# Update PyTorch (CPU)
echo "Updating PyTorch (CPU)..."
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu

# Update dependencies from requirements.txt
echo "Updating dependencies from requirements.txt..."
python -m pip install --upgrade -r requirements.txt

echo ""
echo "========================================"
echo "Update complete! Restarting GUI..."
echo "========================================"
echo ""

# Restart Streamlit on the same port without opening new browser tab
streamlit run gguf_converter/gui.py --server.headless=true --server.address=localhost --server.port=$PORT
