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
Write-Host '  [3] CPU 模式 (无 GPU，使用 MangaOCR)'
Write-Host '  [4] CPU + PaddleOCR 模式 (推荐低配服务器)'
$GPU_CHOICE = Read-Host '请输入选项 [1/2/3/4] (默认 2)'
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
    $env:MOEGAL_USE_GPU = '1'
} elseif ($GPU_CHOICE -eq '3') {
    $REQUIREMENTS_FILE = 'requirements-cpu.txt'
    $TORCH_VERSION = 'torch==2.7.1+cpu torchvision==0.22.1+cpu'
    $TORCH_INDEX_URL = 'https://download.pytorch.org/whl/cpu'
    $env:MOEGAL_USE_GPU = '0'
} elseif ($GPU_CHOICE -eq '4') {
    $REQUIREMENTS_FILE = 'requirements-cpu-paddle.txt'
    $TORCH_VERSION = 'torch==2.7.1+cpu torchvision==0.22.1+cpu'
    $TORCH_INDEX_URL = 'https://download.pytorch.org/whl/cpu'
    $env:MOEGAL_USE_GPU = '0'
    $env:OCR_ENGINE = 'paddle_ocr'
} else {
    $REQUIREMENTS_FILE = 'requirements-cu126.txt'
    $TORCH_VERSION = 'torch==2.7.1+cu126 torchvision==0.22.1+cu126'
    $TORCH_INDEX_URL = 'https://download.pytorch.org/whl/cu126'
    $env:MOEGAL_USE_GPU = '1'
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

Write-Host '正在创建虚拟环境...'
if (-not (Test-Path $VENV_PYTHON)) {
    & $UV_BIN venv --python 3.12
}

Write-Host '正在安装 PyTorch...'
& $UV_BIN pip install $TORCH_VERSION.Split() --extra-index-url $TORCH_INDEX_URL

Write-Host '正在安装其他依赖...'
& $UV_BIN pip install -r $REQUIREMENTS_FILE --index-strategy unsafe-best-match

# 尝试安装 pydensecrf（可选依赖，编译需要 C++ 环境）
Write-Host '尝试安装 pydensecrf (可选: CRF 掩码细化)...'
$pydensecrfResult = & $UV_BIN pip install "pydensecrf@https://github.com/lucasb-eyer/pydensecrf/archive/refs/heads/master.zip" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host '警告: pydensecrf 安装失败，CRF 掩码细化功能将不可用'
    Write-Host '提示: 安装 Visual Studio Build Tools 后可手动安装: pip install pydensecrf'
} else {
    Write-Host 'pydensecrf 安装成功'
}

Write-Host '启动服务...'
if (-not $env:SERVER_PORT) { $env:SERVER_PORT = '8000' }
& $VENV_PYTHON -m uvicorn main:app --host 0.0.0.0 --port $env:SERVER_PORT
