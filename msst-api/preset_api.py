import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from scripts.preset_infer_cli import main as preset_infer  # 你之前整理的 API

app = FastAPI(title="Preset Audio Separation API")

# 跨域（可选）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 临时文件夹，用于上传和处理
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/preset_infer")
async def run_preset_infer(
    preset_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    output_format: str = Form("wav"),
    extra_output_dir: bool = Form(True),
    debug: bool = Form(False)
):
    """
    接收预设 JSON 文件和单个音频文件，运行 preset_infer 分离音频。
    """
    try:
        # 创建唯一处理目录
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(TEMP_UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # 保存上传文件
        preset_path = os.path.join(session_dir, preset_file.filename)
        with open(preset_path, "wb") as f:
            f.write(await preset_file.read())

        input_audio_dir = os.path.join(session_dir, "input")
        os.makedirs(input_audio_dir, exist_ok=True)
        audio_path = os.path.join(input_audio_dir, audio_file.filename)
        with open(audio_path, "wb") as f:
            f.write(await audio_file.read())

        output_dir = os.path.join(session_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # 调用 preset_infer API
        preset_infer(
            preset_path=preset_path,
            input_dir=input_audio_dir,
            output_dir=output_dir,
            output_format=output_format,
            extra_output_dir=extra_output_dir,
            debug=debug
        )

        # 返回输出目录路径（或者可返回音轨列表）
        results = []
        target_dir = os.path.join(output_dir, "extra_output") if extra_output_dir else output_dir
        for file_name in os.listdir(target_dir):
            results.append(os.path.join(target_dir, file_name))

        return JSONResponse({"status": "success", "output_files": results})

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

    finally:
        # 可选：保留 session_dir 或定期清理
        pass


# 启动服务示例
# uvicorn fastapi_preset_api:app --host 0.0.0.0 --port 1145
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_preset_api:app", host="0.0.0.0", port=1145, reload=True)
