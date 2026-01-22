@echo off
setlocal enabledelayedexpansion
REM Run pytest test suite in the project
REM Can be run directly or launched from GUI

cd /d "%~dp0.."

echo Checking for pytest...
venv\Scripts\python.exe -m pytest --version >nul 2>&1
if errorlevel 1 (
    echo pytest is not installed.
    echo.
    set /p "install=Install test dependencies? (y/n): "
    if /i "!install!"=="y" (
        echo.
        echo Installing test dependencies...
        venv\Scripts\python.exe -m pip install -r tests\requirements-dev.txt
        if errorlevel 1 (
            echo.
            echo ERROR: Failed to install test dependencies.
            echo.
            echo Press any key to close...
            pause > nul
            exit /b 1
        )
        echo.
        echo Test dependencies installed successfully.
    ) else (
        echo.
        echo Skipping test dependencies installation.
        echo Cannot run tests without pytest.
        echo.
        echo Press any key to close...
        pause > nul
        exit /b 1
    )
)

echo.
echo Running tests...
echo.
venv\Scripts\python.exe -m pytest
echo.
echo Press any key to close...
pause > nul
