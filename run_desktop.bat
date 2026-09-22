@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0runtime\tts\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] Local runtime was not found.
  echo Run setup.bat first.
  pause
  exit /b 1
)
set PYTHONUTF8=1
"%PY%" desktop_launcher.py
if errorlevel 1 pause
