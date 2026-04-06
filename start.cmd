@echo off
chcp 65001 >nul 2>&1
setlocal EnableExtensions EnableDelayedExpansion

echo Select OCR Engine:
echo   [1] Local OCR
echo   [2] Vision OCR
set /p OCR_CHOICE=Enter option [1/2] (default 1): 
if "%OCR_CHOICE%"=="" set "OCR_CHOICE=1"

echo.
echo Select Running Mode:
echo   [1] RTX 50 series (CUDA 12.8)
echo   [2] Other GPUs (CUDA 12.6)
echo   [3] CPU Mode (No GPU)
set /p GPU_CHOICE=Enter option [1/2/3] (default 2): 
if "%GPU_CHOICE%"=="" set "GPU_CHOICE=2"

if "%OCR_CHOICE%"=="1" set "MOEGAL_OCR_ENGINE=local"
if "%OCR_CHOICE%"=="2" set "MOEGAL_OCR_ENGINE=vision"

if "%GPU_CHOICE%"=="1" (
    set "REQUIREMENTS_FILE=requirements-cu128.txt"
    set "TORCH_VERSION=torch==2.7.1+cu128 torchvision==0.22.1+cu128"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
)
if "%GPU_CHOICE%"=="2" (
    set "REQUIREMENTS_FILE=requirements-cu126.txt"
    set "TORCH_VERSION=torch==2.7.1+cu126 torchvision==0.22.1+cu126"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126"
)
if "%GPU_CHOICE%"=="3" (
    set "REQUIREMENTS_FILE=requirements-cpu.txt"
    set "TORCH_VERSION=torch==2.7.1+cpu torchvision==0.22.1+cpu"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu"
)

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

set "TOOLS_DIR=%ROOT_DIR%.tools"
set "UV_HOME=%TOOLS_DIR%\uv\"
set "VENV_PYTHON=%ROOT_DIR%.venv\Scripts\python.exe"

REM ????????? uv
where uv >nul 2>&1
if %errorlevel%==0 (
    set "UV_BIN=uv"
) else (
    set "UV_BIN=%UV_HOME%uv.exe"
)

set "UV_CACHE_DIR=%ROOT_DIR%.cache\uv"
set "UV_PYTHON_INSTALL_DIR=%ROOT_DIR%.python"
set "UV_PROJECT_ENVIRONMENT=%ROOT_DIR%.venv"
set "UV_PYTHON_PREFERENCE=managed"
set "UV_PYTHON_INSTALL_BIN=0"

if not defined UV_DEFAULT_INDEX set "UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"

if not "%UV_BIN%"=="uv" if not exist "%UV_BIN%" (
    echo Downloading uv...
    if "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
        set "UV_ARCH=aarch64"
    ) else (
        set "UV_ARCH=x86_64"
    )
    set "UV_ZIP_URL=https://github.com/astral-sh/uv/releases/latest/download/uv-!UV_ARCH!-pc-windows-msvc.zip"
    if not exist "%UV_HOME%" mkdir "%UV_HOME%"
    powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing -Uri '!UV_ZIP_URL!' -OutFile \"$env:TEMP\uv.zip\"; Expand-Archive -Path \"$env:TEMP\uv.zip\" -DestinationPath \"$env:TEMP\uv_extract\" -Force; Copy-Item (Get-ChildItem -Path \"$env:TEMP\uv_extract\" -Recurse -Filter 'uv.exe' ^| Select-Object -First 1).FullName '!UV_BIN!' -Force"
)

echo Installing Python 3.12...
"%UV_BIN%" python install 3.12 --no-bin

echo Installing PyTorch...
"%UV_BIN%" pip install %TORCH_VERSION% --extra-index-url %TORCH_INDEX_URL%

echo Installing other dependencies...
"%UV_BIN%" pip install -r "%REQUIREMENTS_FILE%" --index-strategy unsafe-best-match

echo Starting service...
"%VENV_PYTHON%" -m uvicorn main:app --host 0.0.0.0 --port 8000
pause