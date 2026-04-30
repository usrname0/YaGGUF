@echo off
setlocal enabledelayedexpansion

REM Quick launcher for the GUI on Windows

set GPU_BACKEND=

REM Check if venv exists
if not exist venv (
    echo ========================================
    echo Installing YaGGUF
    echo ========================================
    echo.
    echo No existing installation was found. A fresh environment will be created.
    echo Press Ctrl+C to cancel.
    echo.
    echo Select GPU backend for llama.cpp binaries:
    echo   1. CPU only ^(recommended^)
    echo   2. CUDA 12.4 ^(NVIDIA^)
    echo   3. CUDA 13.1 ^(NVIDIA^)
    echo   4. Vulkan ^(NVIDIA/AMD/Intel^)
    echo   5. HIP / ROCm ^(AMD^)
    echo   6. SYCL ^(Intel^)
    echo.
    set GPU_BACKEND_CHOICE=1
    set /p GPU_BACKEND_CHOICE=Enter number [1]:
    if "!GPU_BACKEND_CHOICE!"=="1" set GPU_BACKEND=cpu
    if "!GPU_BACKEND_CHOICE!"=="2" set GPU_BACKEND=cuda-12.4
    if "!GPU_BACKEND_CHOICE!"=="3" set GPU_BACKEND=cuda-13.1
    if "!GPU_BACKEND_CHOICE!"=="4" set GPU_BACKEND=vulkan
    if "!GPU_BACKEND_CHOICE!"=="5" set GPU_BACKEND=hip-radeon
    if "!GPU_BACKEND_CHOICE!"=="6" set GPU_BACKEND=sycl
    if not defined GPU_BACKEND set GPU_BACKEND=cpu
    echo.
    call scripts\setup_windows.bat
    if errorlevel 1 (
        echo.
        echo Setup failed. Please check the errors above.
        pause
        exit /b 1
    )
    echo.
    echo Setup complete! Starting GUI...
    echo.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check and update binaries if needed
echo.
if defined GPU_BACKEND (
    python scripts\check_and_download_binaries.py --backend %GPU_BACKEND%
) else (
    python scripts\check_and_download_binaries.py
)
echo.

echo Starting YaGGUF GUI...
echo.
echo Streamlit will open in your browser automatically.
echo The URL will be displayed below.
echo Press Ctrl+C to stop the server
echo.

streamlit run gguf_converter/gui.py --server.address=localhost
