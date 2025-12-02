@echo off
echo ========================================
echo Yet Another GGUF Converter - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check Python version
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.8 or higher is required
    pause
    exit /b 1
)

echo Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Removing old venv...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing PyTorch (CPU-only version to save space)...
pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo Installing remaining requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo Downloading llama.cpp binaries...
python -m gguf_converter.download_binaries
if errorlevel 1 (
    echo ERROR: Failed to download binaries
    echo You can try running the setup again or install llama.cpp manually
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To use the converter:
echo   - GUI: Run run_gui.bat
echo   - CLI: Run venv\Scripts\activate.bat, then use: python -m gguf_converter
echo.
pause
