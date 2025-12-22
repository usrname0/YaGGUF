@echo off
REM Update dependencies and restart Streamlit

REM Change to project root
cd /d "%~dp0.."

REM Get port from argument or default to 8501
set PORT=%1
if "%PORT%"=="" set PORT=8501

echo.
echo ========================================
echo Updating Dependencies and Restarting
echo ========================================
echo.

REM Kill any process using the specified port
echo Checking for port %PORT%...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% "') do (
    echo Killing process using port %PORT% (PID: %%a^)
    taskkill /F /PID %%a >nul 2>&1
)

REM Give the OS a moment to release the port
timeout /t 2 /nobreak >nul

REM Verify port is free
netstat -ano | findstr ":%PORT% " >nul 2>&1
if %errorlevel% equ 0 (
    echo WARNING: Port %PORT% still in use. Proceeding anyway...
) else (
    echo Port %PORT% is now free
)

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

REM Restart Streamlit on the same port without opening new browser tab
streamlit run gguf_converter/gui.py --server.headless=true --server.address=localhost --server.port=%PORT%
