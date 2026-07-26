# AI Voice API

AI 语音工具 API 封装集合 — 为开源语音项目添加 HTTP API 接口和启动脚本。

## 项目结构

| 模块 | 原始项目 | 新增内容 |
|------|----------|----------|
| `gpt-sovits-api/` | [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) GSVI_v2 分支 | FastAPI TTS 推理接口 (api.py) + 启动脚本 |
| `msst-api/` | [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) | FastAPI 预设音频分离 API (fastapi_preset_api.py) + API 封装层 (preset_api.py) |
| `so-vits-svc-api/` | [So-VITS-SVC](https://github.com/voicevox/so-vits-svc) | Flask 歌声转换 API (flask_api_full_song.py) + 修复补丁 |

## 各模块说明

### GPT-SoVITS API (`gpt-sovits-api/`)

为 GSVI_v2 添加 FastAPI 接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `GET /` | - | 服务状态 |
| `GET /models` | - | 获取可用模型列表 |
| `POST /template` | requestModel | 获取多参考模板 |
| `POST /spks` | requestModel | 获取说话人列表 |
| `POST /infer_ref` | inferWithCustomRefAaudio | 自定义参考音频推理 |
| `POST /infer_single` | inferWithEmotions | 情感推理 |
| `POST /infer_multi` | inferWithMulti | 多人对话推理 |
| `GET /outputs/{path}` | - | 下载生成结果 |

**启动**: `语音合成_CUDA.bat` 或 `语音合成_CPU.bat`

### MSST API (`msst-api/`)

为 MSST-WebUI 添加快速预设 API：

| 接口 | 方法 | 说明 |
|------|------|------|
| `GET /health` | - | 健康检查 |
| `POST /preset_infer` | File+Form | 上传预设+音频进行分离 |
| `GET /presets` | - | 获取预设列表 |
| `POST /infer/local` | File | 本地推理 |

**启动**: `go-webui.bat`

### So-VITS-SVC API (`so-vits-svc-api/`)

为 So-VITS-SVC 添加 Flask API + 修复补丁：

| 接口 | 方法 | 说明 |
|------|------|------|
| `GET /` | - | API 文档页 |
| `GET /health` | - | 健康检查 |
| `POST /wav2wav` | File Upload | 音频文件上传转换 |
| `POST /voiceChangeModel` | Form | 实时变声 |

> 包含 infer_tool 修复补丁（修复 cluster_model_path None 问题、speech_encoder 强制 hubertsoft）

## 配置方法

每个 API 通过**环境变量**定位原始项目路径，无需复制文件到原目录。

推荐目录结构（不强制，路径通过 `.env` 配置）：

```
github/
├── aivoice/            ← 原始语音项目
├── aivoice-api/        ← 本仓库（API 封装）
└── githubres/
```

1. 安装依赖：`pip install -r requirements.txt`
2. 复制 `config/.env.example` 为 `.env`
3. 路径支持**相对路径**（基于脚本所在目录），也可使用绝对路径

### 环境变量说明

| 变量 | 说明 | 适用模块 |
|------|------|----------|
| `GSVI_V2_ROOT` | GSVI_v2 项目根目录 | gpt-sovits-api |
| `MSST_ROOT` | MSST-WebUI 项目根目录 | msst-api |
| `SVC_ROOT` | So-VITS-SVC 项目根目录 | so-vits-svc-api |
| `SVC_MODEL_PATH` | So-VITS-SVC 模型文件路径 | so-vits-svc-api |
| `SVC_CONFIG_PATH` | So-VITS-SVC 配置文件路径 | so-vits-svc-api |
| `CUDA_VISIBLE_DEVICES` | 使用的 GPU 编号 | 全部 |

**注意**：`fastapi_preset_api.py` 和 `preset_api.py` 需要放置在 MSST-WebUI 根目录才能使用 MSST 内部的音频处理引擎，或通过 `MSST_ROOT` 变量指定路径。

## 环境要求

- Python 3.8 - 3.10
- CUDA 11.8+（GPU 推理）
- PyTorch 2.0+

## 更新日志

### 2026-07-26
- 初始化项目结构
- 整理所有自定义 API 封装
