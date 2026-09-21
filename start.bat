@echo off
cd /d "%~dp0"
set FACE_ID_LOCAL=1
echo ========================================================
echo   Starting Face ID Lab Web Application in your browser...
echo ========================================================
python -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application stopped or encountered an error.
    pause
)
