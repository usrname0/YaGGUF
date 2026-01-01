#!/bin/bash
# Update dependencies and restart Streamlit

# Change to project root
cd "$(dirname "$0")/.."

# Get port from argument or default to 8501
PORT=${1:-8501}

echo ""
echo "================================================================================"
echo "                              UPDATE DEPENDENCIES"
echo "                            $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
echo ""

# Kill any process using the specified port
echo "Stopping GUI on port $PORT..."
PID=$(lsof -ti:$PORT 2>/dev/null)
if [ ! -z "$PID" ]; then
    kill -9 $PID 2>/dev/null
    sleep 2
fi

# Activate venv
source venv/bin/activate

# Update PyTorch (CPU)
echo "Updating PyTorch (CPU)..."
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Update dependencies from requirements.txt
echo "Updating dependencies from requirements.txt..."
python -m pip install --upgrade -r requirements.txt

echo ""
echo "Update complete! Restarting GUI..."
echo ""

# Restart Streamlit on the same port without opening new browser tab
streamlit run gguf_converter/gui.py --server.headless=true --server.address=localhost --server.port=$PORT
