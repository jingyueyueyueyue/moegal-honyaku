![](assets/pics/moegal_honyaku.png)

一个日漫图片翻译服务，支持将日文气泡文本自动翻译并回填为中文。

## 项目效果

示例效果：

![example3](./assets/pics/example3.png)

## 技术架构

| 模块 | 技术/模型 |
|------|----------|
| 文字检测 | YOLO (comic-text-segmenter) |
| OCR 引擎 | MangaOCR（日文优先）、PaddleOCR（多语言）、多模态 Vision（GPT-4V/Qwen-VL） |
| 翻译 API | OpenAI 兼容接口、DashScope（通义千问） |
| 图像修复 | OpenCV Inpaint（TELEA/NS 算法） |
| 文本渲染 | PIL + LXGW WenKai 字体 |

### 核心特性

- **双翻译模式**：`parallel` 并发请求每句、`structured` 单次批量翻译
- **多 OCR 引擎**：`local` 本地模型、`vision` 多模态 API
- **GPU 自适应**：CUDA 不可用时自动回退 CPU
- **运行时配置**：无需重启即可切换翻译商/OCR引擎/模式
- **智能掩码**：多策略融合的文字区域掩码生成

## OCR 引擎选择

### local（本地模型）

使用本地 MangaOCR + PaddleOCR 进行识别，适合：
- 高配机器 + 批量处理
- 离线环境 / 隐私敏感场景
- 成本敏感 + 高频调用（无 API 费用）

**注意**：本地模型约 500MB，需要 PyTorch/PaddlePaddle 依赖。

### vision（多模态 OCR）

使用 GPT-4V / Qwen-VL 等多模态模型进行 OCR，**对低配电脑友好**：

| 传统本地 OCR | 多模态 Vision OCR |
|-------------|------------------|
| 需加载 ~500MB+ 模型到内存 | 无需本地模型，零显存占用 |
| 依赖 PyTorch/CUDA 环境 | 仅需 HTTP 请求，CPU 即可运行 |
| 冷启动加载模型需 5-15 秒 | 启动秒级响应 |
| 低配机器 OCR 可能卡顿 2-5 秒/图 | 延迟取决于网络，通常 1-3 秒 |

**优势**：

1. **免模型部署** - 完全卸载到云端，本地零模型负担
2. **更高识别准确率** - 对复杂排版、倾斜文字、模糊图片、手写体适应性更强
3. **多语言原生支持** - 自动识别日/英/中/韩，无需切换引擎
4. **简化依赖** - 仅依赖 OpenAI SDK，安装包体积大幅减小

**适用场景**：
- 低配机器 / 笔记本 / 云服务器
- 追求识别准确率
- 快速体验，不想下载模型

## 项目部署与使用

### 1. 环境准备

- 手动命令行方式：Python `3.12` + `uv`
- Windows `start.cmd` 方式：无需预装 Python/uv

### 2. 安装依赖

在项目根目录执行：

```bash
uv sync
```

### 3. 配置环境变量

在根目录创建 `.env`（或直接修改已有 `.env.example`），至少配置一个可用翻译供应商的 Key：

```env
# OpenAI 方案
OPENAI_API_KEY=your_openai_api_key
# 可选，不填则使用代码默认值
OPENAI_BASE_URL=https://api.openai-proxy.org/v1
OPENAI_MODEL=gpt-5-mini

# DashScope 方案
DASHSCOPE_API_KEY=your_dashscope_api_key
# 可选，不填则使用代码默认值
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3-max

# 多模态 OCR 配置（可选，使用 vision 模式时需要）
VISION_OCR_PROVIDER=openai          # 或 dashscope
VISION_OPENAI_API_KEY=your_key      # 或复用 OPENAI_API_KEY
VISION_OCR_MODEL=gpt-4o-mini        # 或 gpt-4o
```

说明：
- 默认配置为 `openai + parallel`。
- 运行时可通过配置接口切换供应商与翻译模式（见下文）。
- 为了保证首次启动稳定，OCR 默认使用 CPU。若需启用 GPU，可在 `.env` 添加 `MOEGAL_USE_GPU=1`（若驱动/显卡不兼容会自动回退 CPU）。
- 使用 `vision` OCR 模式时，需配置 `VISION_OCR_PROVIDER` 及对应 API Key。

### 4. 启动服务

```bash
uv run uvicorn app.main:app --reload
```

Windows 用户也可直接双击或运行根目录 `start.cmd`：
- 脚本会在项目目录内自动准备 `uv` 与 Python `3.12`。
- 本地运行时目录默认位于 `.tools/`、`.python/`、`.venv/`。

兼容入口也可用：

```bash
uv run uvicorn main:app --reload
```

### 5. 调用接口

#### 5.1 上传图片翻译

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/translate/upload" \
  -F "img=@./assets/pics/example1.png"
```

如不需要返回 `res_img`（base64，体积较大），可关闭：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/translate/upload?include_res_img=false" \
  -F "img=@./assets/pics/example1.png"
```

#### 5.2 通过图片 URL 翻译

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/translate/web" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/xxx.png",
    "referer": "https://example.com",
    "include_res_img": false
  }'
```

#### 5.3 通过 Canvas 导出的 base64 图片翻译

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/translate/web" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "referer": "https://example.com",
    "source_type": "canvas",
    "include_res_img": false
  }'
```

说明：
- `image_url` 与 `image_base64` 必须且只能传一个。
- `image_base64` 当前仅支持完整的 `data:image/png;base64,...` 格式。
- 成功响应中的 `res_img` 仍然是不带 Data URL 前缀的纯 base64 字符串。

#### 5.4 性能相关参数（可选）

可通过环境变量限制 OCR 并发线程数（默认 `2`）：

```env
OCR_MAX_CONCURRENCY=2
```

`/api/v1/translate/web` 的 JSON 请求体默认最大限制为 `20 MiB`，可通过环境变量调整：

```env
TRANSLATE_WEB_MAX_BODY_BYTES=20971520
```

如果服务前面有 Nginx、Caddy 或其他反向代理，需要同步放宽对应的请求体大小限制，否则大尺寸 `canvas` PNG 可能会在到达应用前被代理直接拦截。

#### 5.5 配置接口

```bash
# 初始化为默认配置（openai + parallel）
curl -X POST "http://127.0.0.1:8000/conf/init"

# 查询当前配置
curl "http://127.0.0.1:8000/conf/query"

# 查看可选项
curl "http://127.0.0.1:8000/conf/options"

# 更新配置示例：切换到 dashscope
curl -X POST "http://127.0.0.1:8000/conf/update" \
  -H "Content-Type: application/json" \
  -d '{"attr":"translate_api_type","v":"dashscope"}'

# 更新配置示例：切换 structured 模式
curl -X POST "http://127.0.0.1:8000/conf/update" \
  -H "Content-Type: application/json" \
  -d '{"attr":"translate_mode","v":"structured"}'

# 更新配置示例：切换到 Vision OCR（多模态）
curl -X POST "http://127.0.0.1:8000/conf/update" \
  -H "Content-Type: application/json" \
  -d '{"attr":"ocr_engine","v":"vision"}'

# 更新配置示例：切换回本地 OCR
curl -X POST "http://127.0.0.1:8000/conf/update" \
  -H "Content-Type: application/json" \
  -d '{"attr":"ocr_engine","v":"local"}'
```

### 6. 输出文件

处理后的文件默认保存到：
- `saved/raw/`：原图
- `saved/cn/`：中文回填图

## OCR 场景推荐

| 场景 | 推荐 OCR 引擎 |
|------|--------------|
| 高配机器 + 批量处理 + 无网络限制 | `local`（MangaOCR/PaddleOCR） |
| 低配机器 / 笔记本 / 云服务器 | `vision`（多模态 API） |
| 离线环境 / 隐私敏感 | `local`（完全本地运行） |
| 追求识别准确率 | `vision`（大模型视觉能力更强） |
| 成本敏感 + 高频调用 | `local`（无 API 费用） |

## 项目结构

```text
app/
  api/
    routes/
      manga_translate.py   # 翻译接口
      update_conf.py       # 运行时配置接口
  services/
    ocr.py                 # 模型加载与 OCR
    translate_api.py       # 翻译供应商调用逻辑
    pic_process.py         # 图像处理与文本回填
  core/
    custom_conf.py         # 运行时配置管理
    font_conf.py           # 字体配置
    logger.py              # 日志
    paths.py               # 路径常量
  main.py                  # FastAPI app 创建入口

assets/
  models/                  # 检测模型与 OCR 模型
  fonts/                   # 绘制中文字体
  pics/                    # 示例图片

saved/                     # 输出目录（raw/cn）
logs/                      # 运行日志
main.py                    # 兼容入口（from app.main import app）
```

## 改进计划

- [ ] 添加 Docker 部署支持
- [ ] 批量翻译接口
- [ ] 翻译缓存，避免重复 API 调用
- [ ] 翻译统计与 Token 消耗记录
- [ ] 健康检查接口 `/health`
- [ ] WebSocket 进度通知