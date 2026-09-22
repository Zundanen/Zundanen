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
echo Installing desktop UI component...
"%PY%" -m pip install "pywebview==6.2.1"
if errorlevel 1 (
  echo [ERROR] pywebview installation failed.
  pause
  exit /b 1
)
echo.
echo Desktop UI installed.
echo Double-click: Zundanen.exe
pause
