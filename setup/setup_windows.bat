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

REM Check if virtual environment already exists and is valid
if exist venv\Scripts\activate.bat (
    echo Virtual environment already exists, checking if valid...
    call venv\Scripts\activate.bat

    REM Quick check if streamlit is installed (main dependency)
    python -c "import streamlit" >nul 2>&1
    if errorlevel 1 (
        echo Virtual environment is incomplete, will reinstall packages
        set SKIP_VENV_SETUP=0
    ) else (
        echo Virtual environment is valid, skipping package installation
        set SKIP_VENV_SETUP=1
    )
) else (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create virtual environment
        echo.
        REM Clean up partial venv directory so next run will retry
        if exist venv (
            rmdir /s /q venv
        )
        pause
        exit /b 1
    )

    echo.
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    set SKIP_VENV_SETUP=0
)

if "%SKIP_VENV_SETUP%"=="0" (
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
)

echo.
echo Checking llama.cpp binaries...

REM Check if binaries already exist
python -c "from gguf_converter.binary_manager import BinaryManager; import sys; sys.exit(0 if BinaryManager()._binaries_exist() else 1)" >nul 2>&1
if errorlevel 1 (
    echo Downloading llama.cpp binaries...
    python -m gguf_converter.download_binaries
    if errorlevel 1 (
        echo ERROR: Failed to download binaries
        echo You can try running the setup again or install llama.cpp manually
        pause
        exit /b 1
    )
) else (
    echo Binaries already installed, skipping download
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
