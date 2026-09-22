@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo Zundanen setup
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
  echo.
  echo [ERROR] Setup failed. You can run setup.bat again after fixing the error.
  pause
  exit /b 1
)
echo.
echo Setup completed successfully.
pause
