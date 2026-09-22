@echo off
setlocal
cd /d "%~dp0"

where go.exe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Go was not found.
  echo This script is only for rebuilding the small launcher EXE.
  echo Install Go, then run this file again:
  echo   winget install --id GoLang.Go -e
  echo.
  pause
  exit /b 1
)

if not exist "%~dp0launcher_src\zundanen_launcher.go" (
  echo [ERROR] Launcher source was not found.
  pause
  exit /b 1
)

if not exist "%~dp0dist" mkdir "%~dp0dist"

echo Building small Windows launcher...
set "GOOS=windows"
set "GOARCH=amd64"
set "CGO_ENABLED=0"

go build -trimpath -ldflags="-s -w -H=windowsgui" -o "%~dp0dist\Zundanen.exe" "%~dp0launcher_src\zundanen_launcher.go"
if errorlevel 1 goto :fail

echo.
echo DONE
echo Launcher: %~dp0dist\Zundanen.exe
echo Copy it to the project root when you want to replace the bundled launcher.
echo.
pause
exit /b 0

:fail
echo.
echo [ERROR] Launcher build failed.
pause
exit /b 1
