@echo off
setlocal
cd /d "%~dp0"
title Simple Pothole Image Classifier

echo Simple Pothole Image Classifier
echo.

rem Prefer a Python command that can actually start. A stale Windows
rem Python Launcher entry can exist even when its Python version was removed.
set "PYTHON_CMD="
python -c "import sys" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=python"

if not defined PYTHON_CMD (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD goto :python_missing

rem Repair an incomplete or broken environment from an earlier setup attempt.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo Repairing the previous incomplete setup...
        rmdir /s /q ".venv"
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo First-time setup: creating the virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :setup_failed
)

".venv\Scripts\python.exe" -c "import PIL, tkinterdnd2" >nul 2>&1
if errorlevel 1 (
    echo Installing the required packages. Internet access is needed once...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :setup_failed
)

echo Opening the classifier...
".venv\Scripts\python.exe" app.py
if errorlevel 1 (
    echo.
    echo The application closed because of an error.
    pause
    exit /b 1
)
exit /b 0

:python_missing
echo Python 3 was not found or could not start.
echo Install Python 3 from https://www.python.org/downloads/
echo During installation, select "Add Python to PATH".
echo.
pause
exit /b 1

:setup_failed
echo.
echo Setup failed. Confirm that Python is installed and the internet is available.
echo You can also open README.md for the manual setup commands.
pause
exit /b 1
