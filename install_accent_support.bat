@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_accent_support.ps1"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo Accent support setup finished.
) else (
  echo Accent support setup did not finish. You can run it again safely.
)
echo.
pause
exit /b %RC%
