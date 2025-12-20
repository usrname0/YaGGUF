@echo off
REM Quick launcher for the GUI on Windows

REM Check if venv exists
if not exist venv (
    echo ========================================
    echo Virtual Environment Not Found
    echo ========================================
    echo.
    echo This is your first time running the GUI.
    echo We need to set up the environment first.
    echo.
    echo Press any key to run setup now, or Ctrl+C to cancel...
    pause >nul
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
python scripts\check_binaries.py
echo.

echo Starting GGUF Converter GUI...
echo.
echo Opening browser to http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

streamlit run gguf_converter/gui.py --server.address=localhost
