@echo off
setlocal EnableExtensions

cd /d "%~dp0"

where powershell.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: powershell.exe was not found on this computer.
    echo Please run this project on Windows with PowerShell installed.
    pause
    exit /b 1
)

echo Starting Moegal Honyaku launcher...
set "MOEGAL_LAUNCHED_FROM_CMD=1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Launcher exited with error code %EXIT_CODE%.
    echo Please copy the error messages above when asking for help.
)

pause
exit /b %EXIT_CODE%


