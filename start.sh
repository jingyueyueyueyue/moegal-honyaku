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
echo "  [3] CPU 模式 (无 GPU，使用 MangaOCR)"
echo "  [4] CPU + PaddleOCR 模式 (推荐低配服务器)"
read -p "请输入选项 [1/2/3/4] (默认 2): " GPU_CHOICE
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
    export MOEGAL_USE_GPU=1
elif [ "$GPU_CHOICE" = "3" ]; then
    REQUIREMENTS_FILE="requirements-cpu.txt"
    TORCH_VERSION="torch==2.7.1+cpu torchvision==0.22.1+cpu"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    export MOEGAL_USE_GPU=0
elif [ "$GPU_CHOICE" = "4" ]; then
    REQUIREMENTS_FILE="requirements-cpu-paddle.txt"
    TORCH_VERSION="torch==2.7.1+cpu torchvision==0.22.1+cpu"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    export MOEGAL_USE_GPU=0
    export OCR_ENGINE=paddle_ocr
else
    REQUIREMENTS_FILE="requirements-cu126.txt"
    TORCH_VERSION="torch==2.7.1+cu126 torchvision==0.22.1+cu126"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
    export MOEGAL_USE_GPU=1
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

# 检查虚拟环境是否已存在且可用
if [ -f "$VENV_PYTHON" ]; then
    echo "检测到已有虚拟环境，跳过 Python 安装..."
else
    echo "Installing Python 3.12..."
    # 使用系统 Python 如果可用
    if command -v python3.12 &> /dev/null; then
        export UV_PYTHON_PREFERENCE=system
    fi
    "$UV_BIN" python install 3.12 --no-bin
fi

echo "正在创建虚拟环境..."
if [ ! -f "$VENV_PYTHON" ]; then
    "$UV_BIN" venv --python 3.12
fi

echo "正在安装 PyTorch..."
"$UV_BIN" pip install $TORCH_VERSION --extra-index-url $TORCH_INDEX_URL

echo "正在安装其他依赖..."
"$UV_BIN" pip install -r "$REQUIREMENTS_FILE" --index-strategy unsafe-best-match

# 尝试安装 pydensecrf（可选依赖）
echo "尝试安装 pydensecrf (可选: CRF 掩码细化)..."
if "$UV_BIN" pip install "pydensecrf@https://github.com/lucasb-eyer/pydensecrf/archive/refs/heads/master.zip" 2>/dev/null; then
    echo "pydensecrf 安装成功"
else
    echo "警告: pydensecrf 安装失败，CRF 掩码细化功能将不可用"
    echo "提示: 安装 C++ 编译环境后可手动安装: pip install pydensecrf"
fi

echo "启动服务..."
SERVER_PORT=${SERVER_PORT:-8000}
"$VENV_PYTHON" -m uvicorn main:app --host 0.0.0.0 --port $SERVER_PORT