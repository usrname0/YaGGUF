#!/bin/bash
# Run pytest test suite in the project
# Can be run directly or launched from GUI

# Get script directory and change to project root
cd "$(dirname "$0")/.."

echo "Checking for pytest..."
if ! ./venv/bin/python -m pytest --version >/dev/null 2>&1; then
    echo "pytest is not installed."
    echo ""
    echo -n "Install test dependencies? (y/n): "
    read install
    if [ "$install" = "y" ] || [ "$install" = "Y" ]; then
        echo ""
        echo "Installing test dependencies..."
        ./venv/bin/python -m pip install -r tests/requirements-dev.txt
        if [ $? -ne 0 ]; then
            echo ""
            echo "ERROR: Failed to install test dependencies."
            echo ""
            echo "Press Enter to close..."
            read dummy
            exit 1
        fi
        echo ""
        echo "Test dependencies installed successfully."
    else
        echo ""
        echo "Skipping test dependencies installation."
        echo "Cannot run tests without pytest."
        echo ""
        echo "Press Enter to close..."
        read dummy
        exit 1
    fi
fi

echo ""
echo "Running tests..."
echo ""
./venv/bin/python -m pytest
echo ""
echo "Press Enter to close..."
read dummy
