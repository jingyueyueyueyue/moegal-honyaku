#!/bin/bash

echo "请选择 OCR 引擎："
echo "  [1] 本地 OCR"
echo "  [2] Vision OCR"
read -p "请输入选项 [1/2] (默认 1): " OCR_CHOICE
OCR_CHOICE=${OCR_CHOICE:-1}

echo ""
echo "请选择运行模式："
echo "  [1] RTX 50系列 (CUDA 12.8)"
echo "  [2] 其他显卡 (CUDA 12.6)"
echo "  [3] CPU 模式 (无 GPU)"
read -p "请输入选项 [1/2/3] (默认 2): " GPU_CHOICE
GPU_CHOICE=${GPU_CHOICE:-2}

if [ "$OCR_CHOICE" = "1" ]; then
    export MOEGAL_OCR_ENGINE=local
elif [ "$OCR_CHOICE" = "2" ]; then
    export MOEGAL_OCR_ENGINE=vision
else
    export MOEGAL_OCR_ENGINE=local
fi

if [ "$GPU_CHOICE" = "1" ]; then
    REQUIREMENTS_FILE="requirements-cu128.txt"
    TORCH_VERSION="torch==2.7.1+cu128 torchvision==0.22.1+cu128"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
elif [ "$GPU_CHOICE" = "3" ]; then
    REQUIREMENTS_FILE="requirements-cpu.txt"
    TORCH_VERSION="torch==2.7.1+cpu torchvision==0.22.1+cpu"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
else
    REQUIREMENTS_FILE="requirements-cu126.txt"
    TORCH_VERSION="torch==2.7.1+cu126 torchvision==0.22.1+cu126"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

TOOLS_DIR="$ROOT_DIR/.tools"
UV_HOME="$TOOLS_DIR/uv"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"

# 优先使用全局安装的 uv，否则使用项目本地 uv
if command -v uv &> /dev/null; then
    UV_BIN="uv"
else
    UV_BIN="$UV_HOME/uv"
fi

export UV_CACHE_DIR="$ROOT_DIR/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT_DIR/.python"
export UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv"
export UV_PYTHON_PREFERENCE=managed
export UV_PYTHON_INSTALL_BIN=0

if [ -z "$UV_DEFAULT_INDEX" ]; then
    export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
fi

if [ "$UV_BIN" != "uv" ] && [ ! -f "$UV_BIN" ]; then
    echo "Downloading uv..."
    UV_ARCH=$(uname -m)
    if [ "$UV_ARCH" = "aarch64" ] || [ "$UV_ARCH" = "arm64" ]; then
        UV_ARCH="aarch64"
    else
        UV_ARCH="x86_64"
    fi
    UV_ZIP_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-${UV_ARCH}-unknown-linux-gnu.tar.gz"
    mkdir -p "$UV_HOME"
    UV_TMP=$(mktemp -d)
    curl -fsSL "$UV_ZIP_URL" | tar -xzf - -C "$UV_TMP"
    UV_EXTRACTED=$(find "$UV_TMP" -name "uv" -type f | head -n 1)
    cp "$UV_EXTRACTED" "$UV_BIN"
    chmod +x "$UV_BIN"
    rm -rf "$UV_TMP"
    echo "uv downloaded successfully."
fi

echo "Installing Python 3.12..."
"$UV_BIN" python install 3.12 --no-bin

echo "正在安装 PyTorch..."
"$UV_BIN" pip install $TORCH_VERSION --extra-index-url $TORCH_INDEX_URL

echo "正在安装其他依赖..."
"$UV_BIN" pip install -r "$REQUIREMENTS_FILE" --index-strategy unsafe-best-match

echo "启动服务..."
"$VENV_PYTHON" -m uvicorn main:app --host 0.0.0.0 --port 8000
