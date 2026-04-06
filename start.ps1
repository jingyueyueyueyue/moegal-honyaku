[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host '请选择 OCR 引擎：'
Write-Host '  [1] 本地 OCR'
Write-Host '  [2] Vision OCR'
$OCR_CHOICE = Read-Host '请输入选项 [1/2] (默认 1)'
if ([string]::IsNullOrWhiteSpace($OCR_CHOICE)) { $OCR_CHOICE = '1' }

Write-Host ''
Write-Host '请选择运行模式：'
Write-Host '  [1] RTX 50系列 (CUDA 12.8)'
Write-Host '  [2] 其他显卡 (CUDA 12.6)'
Write-Host '  [3] CPU 模式 (无 GPU)'
$GPU_CHOICE = Read-Host '请输入选项 [1/2/3] (默认 2)'
if ([string]::IsNullOrWhiteSpace($GPU_CHOICE)) { $GPU_CHOICE = '2' }

if ($OCR_CHOICE -eq '1') {
    $env:MOEGAL_OCR_ENGINE = 'local'
} elseif ($OCR_CHOICE -eq '2') {
    $env:MOEGAL_OCR_ENGINE = 'vision'
} else {
    $env:MOEGAL_OCR_ENGINE = 'local'
}

if ($GPU_CHOICE -eq '1') {
    $REQUIREMENTS_FILE = 'requirements-cu128.txt'
    $TORCH_VERSION = 'torch==2.7.1+cu128 torchvision==0.22.1+cu128'
    $TORCH_INDEX_URL = 'https://download.pytorch.org/whl/cu128'
} elseif ($GPU_CHOICE -eq '3') {
    $REQUIREMENTS_FILE = 'requirements-cpu.txt'
    $TORCH_VERSION = 'torch==2.7.1+cpu torchvision==0.22.1+cpu'
    $TORCH_INDEX_URL = 'https://download.pytorch.org/whl/cpu'
} else {
    $REQUIREMENTS_FILE = 'requirements-cu126.txt'
    $TORCH_VERSION = 'torch==2.7.1+cu126 torchvision==0.22.1+cu126'
    $TORCH_INDEX_URL = 'https://download.pytorch.org/whl/cu126'
}

$ROOT_DIR = $PSScriptRoot
Set-Location $ROOT_DIR

$TOOLS_DIR = Join-Path $ROOT_DIR '.tools'
$UV_HOME = Join-Path $TOOLS_DIR 'uv'
$VENV_PYTHON = Join-Path $ROOT_DIR '.venv\Scripts\python.exe'

# 优先使用全局安装的 uv
if (Get-Command uv -ErrorAction SilentlyContinue) {
    $UV_BIN = 'uv'
} else {
    $UV_BIN = Join-Path $UV_HOME 'uv.exe'
}

$env:UV_CACHE_DIR = Join-Path $ROOT_DIR '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ROOT_DIR '.python'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $ROOT_DIR '.venv'
$env:UV_PYTHON_PREFERENCE = 'managed'
$env:UV_PYTHON_INSTALL_BIN = '0'

if (-not $env:UV_DEFAULT_INDEX) {
    $env:UV_DEFAULT_INDEX = 'https://pypi.tuna.tsinghua.edu.cn/simple'
}

if ($UV_BIN -ne 'uv' -and -not (Test-Path $UV_BIN)) {
    Write-Host 'Downloading uv...'
    $UV_ARCH = 'x86_64'
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { $UV_ARCH = 'aarch64' }
    $UV_ZIP_URL = 'https://github.com/astral-sh/uv/releases/latest/download/uv-' + $UV_ARCH + '-pc-windows-msvc.zip'
    $UV_TMP = Join-Path $env:TEMP 'moegal_honyaku_uv.zip'
    $UV_TMP_DIR = Join-Path $env:TEMP 'moegal_honyaku_uv'
    New-Item -ItemType Directory -Force -Path $UV_HOME | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri $UV_ZIP_URL -OutFile $UV_TMP
    Expand-Archive -Path $UV_TMP -DestinationPath $UV_TMP_DIR -Force
    $UV_EXTRACTED = Get-ChildItem -Path $UV_TMP_DIR -Recurse -Filter 'uv.exe' | Select-Object -First 1
    Copy-Item $UV_EXTRACTED.FullName $UV_BIN -Force
    Remove-Item $UV_TMP -Force -ErrorAction SilentlyContinue
    Remove-Item $UV_TMP_DIR -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host 'uv downloaded successfully.'
}

Write-Host 'Installing Python 3.12...'
& $UV_BIN python install 3.12 --no-bin

Write-Host '正在安装 PyTorch...'
& $UV_BIN pip install $TORCH_VERSION.Split() --extra-index-url $TORCH_INDEX_URL

Write-Host '正在安装其他依赖...'
& $UV_BIN pip install -r $REQUIREMENTS_FILE --index-strategy unsafe-best-match

Write-Host '启动服务...'
& $VENV_PYTHON -m uvicorn main:app --host 0.0.0.0 --port 8000
