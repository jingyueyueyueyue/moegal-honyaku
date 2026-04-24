[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = 'Stop'

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host ''
    Write-Host "==> $Message"
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )

    Write-Step $Description
    & $Command
    $exitCode = $LASTEXITCODE
    if ($null -ne $exitCode -and $exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode."
    }
}

function New-FilteredRequirementsFile {
    param([Parameter(Mandatory = $true)][string]$SourcePath)

    $targetPath = Join-Path $env:TEMP 'moegal_honyaku_requirements_no_torch.txt'
    $skipPattern = '^(\s*(--extra-index-url|--index-url|--find-links)\b|\s*(torch|torchvision)\s*[=<>!~@])'
    Get-Content -LiteralPath $SourcePath | Where-Object { $_ -notmatch $skipPattern } | Set-Content -LiteralPath $targetPath -Encoding UTF8
    return $targetPath
}

function Assert-PythonModule {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$ModuleName
    )

    & $PythonPath -c "import $ModuleName" 2>$null
    $exitCode = $LASTEXITCODE
    if ($null -ne $exitCode -and $exitCode -ne 0) {
        throw "Required Python module '$ModuleName' is missing. Dependency installation did not complete."
    }
}

function Select-Options {
    Write-Host 'Select OCR Engine:'
    Write-Host '  [1] Local OCR'
    Write-Host '  [2] Vision OCR'
    $ocrChoice = Read-Host 'Enter option [1/2] (default 1)'
    if ([string]::IsNullOrWhiteSpace($ocrChoice)) { $ocrChoice = '1' }

    switch ($ocrChoice) {
        '1' { $env:MOEGAL_OCR_ENGINE = 'local' }
        '2' { $env:MOEGAL_OCR_ENGINE = 'vision' }
        default {
            Write-Host "Invalid OCR option '$ocrChoice'. Using Local OCR."
            $env:MOEGAL_OCR_ENGINE = 'local'
        }
    }

    Write-Host ''
    Write-Host 'Select Mode:'
    Write-Host '  [1] RTX 50 Series (CUDA 12.8)'
    Write-Host '  [2] Other GPU (CUDA 12.6)'
    Write-Host '  [3] CPU Mode (No GPU, MangaOCR)'
    Write-Host '  [4] CPU + PaddleOCR Mode (recommended for low-spec machines)'
    $gpuChoice = Read-Host 'Enter option [1/2/3/4] (default 2)'
    if ([string]::IsNullOrWhiteSpace($gpuChoice)) { $gpuChoice = '2' }

    switch ($gpuChoice) {
        '1' {
            return @{
                RequirementsFile = 'requirements-cu128.txt'
                TorchPackages = @('torch==2.7.1+cu128', 'torchvision==0.22.1+cu128')
                TorchIndexUrl = 'https://download.pytorch.org/whl/cu128'
                UseGpu = '1'
            }
        }
        '3' {
            return @{
                RequirementsFile = 'requirements-cpu.txt'
                TorchPackages = @('torch==2.7.1+cpu', 'torchvision==0.22.1+cpu')
                TorchIndexUrl = 'https://download.pytorch.org/whl/cpu'
                UseGpu = '0'
            }
        }
        '4' {
            $env:OCR_ENGINE = 'paddle_ocr'
            return @{
                RequirementsFile = 'requirements-cpu-paddle.txt'
                TorchPackages = @('torch==2.7.1+cpu', 'torchvision==0.22.1+cpu')
                TorchIndexUrl = 'https://download.pytorch.org/whl/cpu'
                UseGpu = '0'
            }
        }
        default {
            if ($gpuChoice -ne '2') {
                Write-Host "Invalid mode option '$gpuChoice'. Using Other GPU (CUDA 12.6)."
            }
            return @{
                RequirementsFile = 'requirements-cu126.txt'
                TorchPackages = @('torch==2.7.1+cu126', 'torchvision==0.22.1+cu126')
                TorchIndexUrl = 'https://download.pytorch.org/whl/cu126'
                UseGpu = '1'
            }
        }
    }
}

try {
    $options = Select-Options
    $env:MOEGAL_USE_GPU = $options.UseGpu

    $rootDir = $PSScriptRoot
    if ([string]::IsNullOrWhiteSpace($rootDir)) {
        $rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    Set-Location $rootDir

    $requirementsPath = Join-Path $rootDir $options.RequirementsFile
    if (-not (Test-Path -LiteralPath $requirementsPath)) {
        throw "Requirements file not found: $requirementsPath"
    }

    $toolsDir = Join-Path $rootDir '.tools'
    $uvHome = Join-Path $toolsDir 'uv'
    $venvPython = Join-Path $rootDir '.venv\Scripts\python.exe'

    $globalUv = Get-Command uv -ErrorAction SilentlyContinue
    if ($globalUv) {
        $uvBin = $globalUv.Source
    } else {
        $uvBin = Join-Path $uvHome 'uv.exe'
    }

    $env:UV_CACHE_DIR = Join-Path $rootDir '.cache\uv'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $rootDir '.python'
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $rootDir '.venv'
    $env:UV_PYTHON_PREFERENCE = 'managed'
    $env:UV_PYTHON_INSTALL_BIN = '0'

    if (-not $env:UV_DEFAULT_INDEX) {
        $env:UV_DEFAULT_INDEX = 'https://pypi.tuna.tsinghua.edu.cn/simple'
    }

    if (-not (Test-Path -LiteralPath $uvBin)) {
        Write-Step 'Downloading uv'
        $uvArch = 'x86_64'
        if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { $uvArch = 'aarch64' }
        $uvZipUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-$uvArch-pc-windows-msvc.zip"
        $uvTmp = Join-Path $env:TEMP 'moegal_honyaku_uv.zip'
        $uvTmpDir = Join-Path $env:TEMP 'moegal_honyaku_uv'

        New-Item -ItemType Directory -Force -Path $uvHome | Out-Null
        Remove-Item $uvTmp -Force -ErrorAction SilentlyContinue
        Remove-Item $uvTmpDir -Recurse -Force -ErrorAction SilentlyContinue

        Invoke-WebRequest -UseBasicParsing -Uri $uvZipUrl -OutFile $uvTmp
        Expand-Archive -Path $uvTmp -DestinationPath $uvTmpDir -Force
        $uvExtracted = Get-ChildItem -Path $uvTmpDir -Recurse -Filter 'uv.exe' | Select-Object -First 1
        if (-not $uvExtracted) {
            throw 'uv.exe was not found in the downloaded archive.'
        }
        Copy-Item $uvExtracted.FullName $uvBin -Force
        Remove-Item $uvTmp -Force -ErrorAction SilentlyContinue
        Remove-Item $uvTmpDir -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "uv downloaded to: $uvBin"
    }

    Invoke-Checked 'Installing Python 3.12' { & $uvBin python install 3.12 --no-bin }

    if (-not (Test-Path -LiteralPath $venvPython)) {
        Invoke-Checked 'Creating virtual environment' { & $uvBin venv --python 3.12 }
    } else {
        Write-Step 'Virtual environment already exists'
        Write-Host $venvPython
    }

    Invoke-Checked 'Installing PyTorch' { & $uvBin pip install @($options.TorchPackages) --index-url $options.TorchIndexUrl }

    $filteredRequirementsPath = New-FilteredRequirementsFile -SourcePath $requirementsPath
    Invoke-Checked 'Installing dependencies' { & $uvBin pip install -r $filteredRequirementsPath --index-url $env:UV_DEFAULT_INDEX --index-strategy unsafe-best-match }

    Write-Step 'Installing optional pydensecrf'
    & $uvBin pip install 'pydensecrf@https://github.com/lucasb-eyer/pydensecrf/archive/refs/heads/master.zip'
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'WARNING: pydensecrf installation failed. CRF mask refinement will be unavailable.'
        Write-Host 'TIP: Install Visual Studio Build Tools, then run: pip install pydensecrf'
    } else {
        Write-Host 'pydensecrf installed successfully.'
    }

    if (-not (Test-Path -LiteralPath $venvPython)) {
        throw "Python executable not found after setup: $venvPython"
    }
    Assert-PythonModule -PythonPath $venvPython -ModuleName 'uvicorn'

    if (-not $env:SERVER_PORT) { $env:SERVER_PORT = '8000' }
    Write-Step "Starting server on port $env:SERVER_PORT"
    & $venvPython -m uvicorn main:app --host 0.0.0.0 --port $env:SERVER_PORT
    $serverExitCode = $LASTEXITCODE
    if ($null -ne $serverExitCode -and $serverExitCode -ne 0) {
        throw "Server exited with code $serverExitCode."
    }
}
catch {
    Write-Host ''
    Write-Host 'ERROR: Startup failed.' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ''
    Write-Host 'Common fixes:'
    Write-Host '  1. Check network access to GitHub, PyPI, and download.pytorch.org.'
    Write-Host '  2. Try CPU mode if the target computer has no NVIDIA GPU or incompatible drivers.'
    Write-Host '  3. Make sure the project path is not inside a protected system directory.'
    Write-Host '  4. Copy all project files, including requirements-*.txt, .env, app, models, and assets.'
    exit 1
}
finally {
    Write-Host ''
    if ($env:MOEGAL_LAUNCHED_FROM_CMD -ne '1') {
        Write-Host 'Press Enter to exit...'
        [void][System.Console]::ReadLine()
    }
}

