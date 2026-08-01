# AI Voice API

AI 语音工具 API 封装集合 — 为开源语音项目添加 HTTP API 接口和启动脚本。

## 项目结构

| 模块 | 原始项目 | 新增内容 |
|------|----------|----------|
| `gpt-sovits-api/` | [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) GSVI_v2 分支 | FastAPI TTS 推理接口 (api.py) + 启动脚本 |
| `msst-api/` | [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) | FastAPI 预设音频分离 API (fastapi_preset_api.py) + API 封装层 (preset_api.py) |
| `so-vits-svc-api/` | [So-VITS-SVC](https://github.com/voicevox/so-vits-svc) | Flask 歌声转换 API (flask_api_full_song.py) + 修复补丁 |
| `flow-web/` | 本仓库自研 | 分离 → 变声 全流程 Web 界面（自动拉起 MSST / SVC 服务） |

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

### Flow-Web 全流程处理 (`flow-web/`)

基于本地配置一键处理 Web 界面：上传音频 → MSST 人声/伴奏分离 → SVC 变声 → 试听/下载，自动拉起 MSST 与 SVC 服务。

| 功能 | 说明 |
|------|------|
| 服务面板 | 查看/启动/停止 MSST、SVC 状态与模型/说话人选择 |
| 滑块式 SVC 参数 | 变调、切片长度、切片阈值、接口淡化等 12 项滑动调节 + F0/扩散开关，实时数值显示 |
| 处理前滑动切片 | 波形可视化，拖拽选择只处理音频区间（ffmpeg 裁剪） |
| 试听 | 上传后试听原音频，完成后试听结果/各分轨 |
| SVC 预设 | 保存/加载 SVC 参数组合到 `svc_presets/` |
| 合并伴奏 | 处理后音频 + 伴奏以 ffmpeg amix 合并，人声/伴奏音量独立调节 |

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/process` | POST | 全流程处理（分离 + 变声，支持 start_time/end_time 切片） |
| `/api/task/{id}` | GET | 任务进度查询 |
| `/api/download/{session}/{file}` | GET | 下载处理结果 |
| `/api/services` | GET | 服务健康状态 |
| `/api/services/{name}/start` `stop` | POST | 启停 MSST / SVC |
| `/api/models` `speakers` `svc_state` | GET | 模型/说话人查询 |
| `/api/svc_reload` | POST | 动态重载 SVC 模型 |
| `/api/svc_presets` `save` `load` | GET/POST | SVC 预设管理 |
| `/api/mix` | POST | 合并处理后音频与伴奏 |

**启动**: `start.bat`（自动拉起 MSST / SVC），浏览器访问 http://127.0.0.1:8010

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

## 说明

本仓库中的部分代码由 AI 辅助编写（AI 生成的代码），**仅用于测试用途**，请勿用于生产环境。

## 更新日志

### 2026-07-26
- 初始化项目结构
- 整理所有自定义 API 封装

### 2026-08-01
- 新增 `flow-web/` 全流程处理 Web：滑块式 SVC 参数面板、处理前滑动切片（波形选区）、输入/结果试听、SVC 预设保存加载、ffmpeg 合并伴奏
- 修复 `flow-web` JSONResponse 参数顺序错误（所有错误路径可正常返回）
- SVC API 全功能：自动切片（clip_seconds/pad_seconds）、接口淡化（lg_num/lgr_num）、扩散（k_step）、增强（enhancer_adaptive_key）、动态模型重载
- MSST 伴奏轨修复：karaoke 模型 stems 为 `Instrumental`，输出文件按 `_accompaniment` 等命名规约保存
