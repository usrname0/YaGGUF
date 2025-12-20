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
echo Waiting for Streamlit to shut down...
:CHECK_PORT
netstat -ano | findstr ":8501" >nul 2>&1
if %errorlevel% equ 0 (
    timeout /t 1 /nobreak >nul
    goto CHECK_PORT
)
echo Port 8501 is free

REM Activate venv
call venv\Scripts\activate.bat

REM Update dependencies from requirements.txt
echo Updating dependencies from requirements.txt...
python -m pip install --upgrade -r requirements.txt

echo.
echo ========================================
echo Update complete! Restarting GUI...
echo ========================================
echo.

REM Restart Streamlit without opening new browser tab
streamlit run gguf_converter/gui.py --server.headless=true
