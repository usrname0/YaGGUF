@echo off
REM Minimal setup script for Windows
REM This script is called by run_gui.bat and should not be run directly.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH. Please install Python 3.8 or higher.
    exit /b 1
)

REM Check Python version
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.8 or higher is required.
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    REM Clean up partial venv directory so next run will retry
    if exist venv (
        rmdir /s /q venv
    )
    exit /b 1
)

REM Activate virtual environment so pip installs into it
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch...
pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch.
    exit /b 1
)

echo Installing requirements...

pip install -r requirements.txt

if errorlevel 1 (

    echo ERROR: Failed to install requirements.

    exit /b 1

)



exit /b 0
