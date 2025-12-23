@echo off
REM Update YaGUFF to latest version and restart Streamlit

REM Change to project root
cd /d "%~dp0.."

REM Get port from argument or default to 8501
set PORT=%1
if "%PORT%"=="" set PORT=8501

REM Get version from second argument (optional)
set VERSION=%2

echo.
echo ========================================
echo Updating YaGUFF and Restarting
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

REM Update YaGUFF
echo Fetching latest YaGUFF version...
git fetch --tags

if "%VERSION%"=="" (
    echo Error: Version not specified
    pause
    exit /b 1
)

echo Checking out version %VERSION%...
git checkout %VERSION%

if %errorlevel% neq 0 (
    echo Error: Failed to checkout version %VERSION%
    pause
    exit /b 1
)

echo.
echo ========================================
echo Update complete! Restarting GUI...
echo ========================================
echo.

REM Activate venv if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Restart Streamlit on the same port without opening new browser tab
streamlit run gguf_converter/gui.py --server.headless=true --server.address=localhost --server.port=%PORT%
