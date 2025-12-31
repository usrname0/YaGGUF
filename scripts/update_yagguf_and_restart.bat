@echo off
REM Update YaGGUF to latest version and restart Streamlit

REM Change to project root
cd /d "%~dp0.."

REM Get port from argument or default to 8501
set PORT=%1
if "%PORT%"=="" set PORT=8501

REM Get version from second argument (optional)
set VERSION=%2

echo.
echo ========================================
echo Updating YaGGUF and Restarting
echo ========================================
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

REM Update YaGGUF
echo Fetching latest YaGGUF version...
git fetch --tags 2>nul

if "%VERSION%"=="" (
    echo Error: Version not specified
    pause
    exit /b 1
)

echo Checking out version %VERSION%...
git -c advice.detachedHead=false checkout %VERSION%

if errorlevel 1 (
    echo.
    echo Error: Failed to checkout version %VERSION%
    echo.
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
