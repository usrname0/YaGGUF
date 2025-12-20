@echo off
REM Update dependencies and restart Streamlit

REM Change to project root
cd /d "%~dp0.."

echo.
echo ========================================
echo Updating Dependencies and Restarting
echo ========================================
echo.

REM Wait for port 8501 to be released
echo|set /p="Waiting for port 8501 to be free"
:CHECK_PORT
netstat -ano | findstr ":8501" >nul 2>&1
if %errorlevel% equ 0 (
    echo|set /p="."
    timeout /t 1 /nobreak >nul
    goto CHECK_PORT
)
echo.
echo Port 8501 is free

REM Activate venv
call venv\Scripts\activate.bat

REM Update PyTorch (CPU)
echo Updating PyTorch (CPU)...
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu

REM Update dependencies from requirements.txt
echo Updating dependencies from requirements.txt...
python -m pip install --upgrade -r requirements.txt

echo.
echo ========================================
echo Update complete! Restarting GUI...
echo ========================================
echo.

REM Restart Streamlit without opening new browser tab
streamlit run gguf_converter/gui.py --server.headless=true --server.address=localhost
