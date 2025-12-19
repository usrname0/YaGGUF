@echo off
REM Update dependencies and restart Streamlit

echo.
echo ========================================
echo Updating Dependencies and Restarting
echo ========================================
echo.

REM Wait for Streamlit to fully exit
timeout /t 2 /nobreak >nul

REM Activate venv
call venv\Scripts\activate.bat

REM Update all dependencies
echo Updating dependencies from requirements.txt...
python -m pip install --upgrade -r requirements.txt

echo.
echo ========================================
echo Update complete! Restarting GUI...
echo ========================================
echo.

REM Restart Streamlit
streamlit run gguf_converter/gui.py
