@echo off
REM Update dependencies and restart Streamlit

REM Change to project root
cd /d "%~dp0.."

REM Get port from argument or default to 8501
set PORT=%1
if "%PORT%"=="" set PORT=8501

echo.
echo ================================================================================
echo                              UPDATE DEPENDENCIES
echo                                %DATE% %TIME:~0,8%
echo ================================================================================
echo.

REM Kill any process using the specified port
echo Stopping GUI on port %PORT%...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    if not "%%a"=="0" (
        taskkill /F /PID %%a >nul 2>&1
    )
)

REM Give the OS a moment to release the port
timeout /t 2 /nobreak >nul

REM Activate venv
call venv\Scripts\activate.bat

REM Update PyTorch (CPU)
echo Updating PyTorch (CPU)...
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Update dependencies from requirements.txt
echo Updating dependencies from requirements.txt...
python -m pip install --upgrade -r requirements.txt

echo.
echo Update complete! Restarting GUI...
echo.

REM Restart Streamlit on the same port without opening new browser tab
streamlit run gguf_converter/gui.py --server.headless=true --server.address=localhost --server.port=%PORT%
