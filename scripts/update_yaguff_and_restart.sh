#!/bin/bash
# Update YaGUFF to latest version and restart Streamlit

# Change to project root
cd "$(dirname "$0")/.."

# Get port from argument or default to 8501
PORT=${1:-8501}

# Get version from second argument
VERSION=$2

echo ""
echo "========================================"
echo "Updating YaGUFF and Restarting"
echo "========================================"
echo ""

# Kill any process using the specified port
echo "Stopping GUI on port $PORT..."
PID=$(lsof -ti:$PORT 2>/dev/null)
if [ ! -z "$PID" ]; then
    kill -9 $PID 2>/dev/null
    sleep 2
fi

# Update YaGUFF
echo "Fetching latest YaGUFF version..."
git fetch --tags 2>/dev/null

if [ -z "$VERSION" ]; then
    echo "Error: Version not specified"
    exit 1
fi

echo "Checking out version $VERSION..."
git -c advice.detachedHead=false checkout $VERSION 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Error: Failed to checkout version $VERSION"
    exit 1
fi

echo ""
echo "========================================"
echo "Update complete! Restarting GUI..."
echo "========================================"
echo ""

# Activate venv if it exists
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

# Restart Streamlit on the same port without opening new browser tab
streamlit run gguf_converter/gui.py --server.headless=true --server.address=localhost --server.port=$PORT
