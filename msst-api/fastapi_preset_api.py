import os,sys,glob,json,shutil,uuid
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.environ.get("MSST_ROOT")
if ROOT_DIR:
    ROOT_DIR = os.path.abspath(os.path.join(_repo_root, ROOT_DIR))
else:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from scripts.preset_api import preset_infer

app = FastAPI(title="Preset Audio Separation API")

# 跨域设置（可选）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 临时文件夹
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

PRESET_DIR = os.path.join(ROOT_DIR, "presets")
os.makedirs(PRESET_DIR, exist_ok=True)

@app.get("/presets")
async def get_presets():
    """
    获取可用的预设列表
    """
    try:
        # 查找所有 JSON 预设文件
        preset_files = glob.glob(os.path.join(PRESET_DIR, "*.json"))
        presets = []
        
        for preset_file in preset_files:
            try:
                with open(preset_file, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)
                    
                presets.append({
                    "name": preset_data.get("name", os.path.basename(preset_file)),
                    "filename": os.path.basename(preset_file),
                    "description": preset_data.get("description", ""),
                    "version": preset_data.get("version", "1.0"),
                    "filepath": preset_file
                })
            except Exception as e:
                print(f"读取预设文件失败 {preset_file}: {e}")
                # 仍然返回文件名，即使读取内容失败
                presets.append({
                    "name": os.path.basename(preset_file),
                    "filename": os.path.basename(preset_file),
                    "description": "读取预设信息失败",
                    "version": "1.0",
                    "filepath": preset_file
                })
        
        return {
            "status": "success",
            "count": len(presets),
            "presets": presets
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"获取预设列表失败: {str(e)}"
            }
        )

@app.get("/presets/{preset_name}")
async def get_preset(preset_name: str):
    """
    获取特定预设的详细信息
    """
    try:
        preset_path = os.path.join(PRESET_DIR, preset_name)
        
        if not os.path.exists(preset_path):
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"预设文件不存在: {preset_name}"
                }
            )
        
        with open(preset_path, 'r', encoding='utf-8') as f:
            preset_data = json.load(f)
        
        return {
            "status": "success",
            "preset": preset_data,
            "filename": preset_name
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"读取预设失败: {str(e)}"
            }
        )

@app.post("/presets/upload")
async def upload_preset(
    preset_file: UploadFile = File(...),
    overwrite: bool = Form(False)
):
    """
    上传新的预设文件
    """
    try:
        filename = preset_file.filename
        
        # 确保是 JSON 文件
        if not filename.lower().endswith('.json'):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "只能上传 JSON 文件"
                }
            )
        
        preset_path = os.path.join(PRESET_DIR, filename)
        
        # 检查文件是否已存在
        if os.path.exists(preset_path) and not overwrite:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"文件已存在: {filename}。使用 overwrite=true 覆盖"
                }
            )
        
        # 保存文件
        content = await preset_file.read()
        with open(preset_path, 'wb') as f:
            f.write(content)
        
        return {
            "status": "success",
            "message": f"预设上传成功: {filename}",
            "filepath": preset_path
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"上传预设失败: {str(e)}"
            }
        )

@app.post("/infer/local")
async def infer_local(
    request: Request,
    preset_file: UploadFile = File(None),
    audio_file: UploadFile = File(None),
    input_file: UploadFile = File(None),
    preset_path: str = Form(None),
    audio_path: str = Form(None),
    output_format: str = Form("wav"),
    extra_output_dir: bool = Form(True),
    debug: bool = Form(False),
    use_tta: bool = Form(False),
    force_cpu: bool = Form(False)
):
    """
    支持插件调用 /infer/local
    """
    try:
        # 创建临时会话目录
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(TEMP_UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        print(f"Starting inference for session: {session_id}")

        # 处理预设
        preset_path_to_use = None
        
        # 优先处理预设文件上传
        if preset_file:
            preset_path_to_use = os.path.join(session_dir, "preset.json")
            with open(preset_path_to_use, "wb") as f:
                f.write(await preset_file.read())
        elif preset_path and os.path.exists(preset_path):
            preset_path_to_use = preset_path
        else:
            # 从表单数据中查找预设
            form_data = await request.form()
            if 'preset_path' in form_data:
                preset_path_to_use = form_data['preset_path']
            else:
                return {"error": "preset_file or preset_path required"}

        # 处理音频文件 - 插件使用 input_file 字段
        input_dir = os.path.join(session_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        
        # 确定音频文件来源
        audio_source = None
        if input_file:  # 插件使用 input_file
            audio_source = input_file
        elif audio_file:
            audio_source = audio_file
        
        if audio_source:
            audio_file_path = os.path.join(input_dir, audio_source.filename)
            with open(audio_file_path, "wb") as f:
                f.write(await audio_source.read())
        elif audio_path and os.path.exists(audio_path):
            # 使用音频文件路径
            audio_file_path = os.path.join(input_dir, os.path.basename(audio_path))
            shutil.copy(audio_path, audio_file_path)
        else:
            return {"error": "audio_file or audio_path required"}

        # 输出目录
        output_dir = os.path.join(session_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        print(f"Calling CLI with: input_folder={input_dir}, store_dir={output_dir}, preset_path={preset_path_to_use}")

        # 调用 CLI 主逻辑
        from scripts.preset_infer_cli import main as cli_main
        cli_main(
            input_folder=input_dir,
            store_dir=output_dir,
            preset_path=preset_path_to_use,
            output_format=output_format,
            extra_output_dir=extra_output_dir
        )

        print(f"Inference completed. Checking output directory: {output_dir}")
        
        # 查找输出文件
        target_dir = os.path.join(output_dir, "extra_output") if extra_output_dir else output_dir
        
        # 确保目标目录存在
        if not os.path.exists(target_dir):
            print(f"Target directory {target_dir} does not exist!")
            target_dir = output_dir  # 回退到 output 目录
        
        print(f"Searching for files in: {target_dir}")
        
        files = []
        if os.path.exists(target_dir):
            for f in os.listdir(target_dir):
                if f.lower().endswith(('.wav', '.mp3', '.flac')):
                    print(f"Found file: {f}")
                    files.append({
                        "name": f,
                        "url": f"http://127.0.0.1:9000/download/{session_id}/{f}",
                        "simple_url": f"http://127.0.0.1:9000/download/{f}",
                        "session_id": session_id
                    })
        else:
            print(f"Directory {target_dir} does not exist. Available in session_dir:")
            if os.path.exists(session_dir):
                print(f"Session dir contents: {os.listdir(session_dir)}")
        
        # 如果没找到文件，检查 output 目录本身
        if not files and os.path.exists(output_dir):
            print(f"Checking output directory directly: {output_dir}")
            for f in os.listdir(output_dir):
                if f.lower().endswith(('.wav', '.mp3', '.flac')):
                    print(f"Found file in output dir: {f}")
                    files.append({
                        "name": f,
                        "url": f"http://127.0.0.1:9000/download/{session_id}/{f}",
                        "simple_url": f"http://127.0.0.1:9000/download/{f}",
                        "session_id": session_id
                    })
        
        print(f"Returning {len(files)} files")
        
        return {
            "status": "success", 
            "session_id": session_id,
            "files": files,
            "output_dir": output_dir,
            "target_dir": target_dir,
            "file_count": len(files)
        }

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in infer_local: {e}\n{error_traceback}")
        
        return JSONResponse(
            status_code=500, 
            content={
                "status": "error", 
                "message": str(e),
                "traceback": error_traceback
            }
        )

@app.get("/")
async def root():
    """
    根端点，显示API信息
    """
    endpoints = [
        {"method": "POST", "path": "/infer/local", "description": "处理音频文件（插件兼容）"},
        {"method": "POST", "path": "/infer/plugin", "description": "处理音频文件（专门为插件设计）"},
        {"method": "GET", "path": "/list_outputs", "description": "获取处理结果，参数：session_id 或 preset_name"},
        {"method": "GET", "path": "/download/{session_id}/{filename}", "description": "下载文件"},
        {"method": "GET", "path": "/presets", "description": "获取预设列表"},
        {"method": "GET", "path": "/presets/{preset_name}", "description": "获取预设详情"},
        {"method": "POST", "path": "/presets/upload", "description": "上传预设"},
        {"method": "GET", "path": "/health", "description": "健康检查"}
    ]
    
    # 统计临时文件
    temp_stats = {}
    if os.path.exists(TEMP_UPLOAD_DIR):
        sessions = [d for d in os.listdir(TEMP_UPLOAD_DIR) 
                   if os.path.isdir(os.path.join(TEMP_UPLOAD_DIR, d))]
        temp_stats = {
            "session_count": len(sessions),
            "sessions": sessions[:10]  # 只显示前10个
        }
    
    return {
        "service": "Preset Audio Separation API",
        "version": "1.0",
        "endpoints": endpoints,
        "temp_uploads": temp_stats,
        "note": "Use /infer/local for plugin compatibility"
    }

@app.get("/list_outputs")
async def list_outputs(
    session_id: str = None, 
    preset_name: str = None,
    preset_path: str = None,
    request: Request = None
):
    """
    获取处理结果文件列表
    """
    try:
        print(f"list_outputs called with session_id={session_id}, preset_name={preset_name}, preset_path={preset_path}")
        
        # 尝试从查询参数获取
        if not session_id and not preset_name and not preset_path:
            # 尝试从请求URL参数获取
            if request:
                query_params = dict(request.query_params)
                print(f"Query params: {query_params}")
                if 'session_id' in query_params:
                    session_id = query_params['session_id']
                elif 'preset_name' in query_params:
                    preset_name = query_params['preset_name']
                elif 'preset_path' in query_params:
                    preset_path = query_params['preset_path']
        
        if not session_id:
            # 查找最新的会话
            latest_session = None
            latest_time = 0
            
            if os.path.exists(TEMP_UPLOAD_DIR):
                for session in os.listdir(TEMP_UPLOAD_DIR):
                    session_dir = os.path.join(TEMP_UPLOAD_DIR, session)
                    if os.path.isdir(session_dir):
                        # 检查是否有 output 目录
                        output_dir = os.path.join(session_dir, "output")
                        if os.path.exists(output_dir):
                            dir_time = os.path.getmtime(output_dir)
                            if dir_time > latest_time:
                                latest_time = dir_time
                                latest_session = session
            
            if latest_session:
                session_id = latest_session
                print(f"Using latest session: {session_id}")
            else:
                return {
                    "status": "error",
                    "message": "No session found and no session_id provided"
                }
        
        session_dir = os.path.join(TEMP_UPLOAD_DIR, session_id)
        print(f"Looking for session dir: {session_dir}")
        
        if not os.path.exists(session_dir):
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"Session not found: {session_id}",
                    "available_sessions": os.listdir(TEMP_UPLOAD_DIR) if os.path.exists(TEMP_UPLOAD_DIR) else []
                }
            )
        
        # 首先检查 output 目录
        output_dir = os.path.join(session_dir, "output")
        print(f"Looking for output dir: {output_dir}")
        
        if not os.path.exists(output_dir):
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"Output directory not found for session: {session_id}",
                    "session_dir_contents": os.listdir(session_dir) if os.path.exists(session_dir) else []
                }
            )
        
        # 查找 output 目录及其子目录中的所有音频文件
        files = []
        for root, dirs, filenames in os.walk(output_dir):
            print(f"Searching in directory: {root}")
            for filename in filenames:
                if filename.lower().endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, output_dir)
                    
                    print(f"Found audio file: {filename} at {file_path}")
                    
                    files.append({
                        "name": filename,
                        "relative_path": rel_path,
                        "url": f"http://127.0.0.1:9000/download/{session_id}/{rel_path}",
                        "simple_url": f"http://127.0.0.1:9000/download/{filename}",
                        "session_id": session_id,
                        "size": os.path.getsize(file_path)
                    })
        
        print(f"Total files found: {len(files)}")
        
        if not files:
            # 如果没找到文件，列出 output 目录的内容以便调试
            output_contents = []
            if os.path.exists(output_dir):
                output_contents = os.listdir(output_dir)
                print(f"Output directory contents: {output_contents}")
                
                # 也尝试查找可能存在的子目录
                for item in output_contents:
                    item_path = os.path.join(output_dir, item)
                    if os.path.isdir(item_path):
                        print(f"Subdirectory {item} contents: {os.listdir(item_path)}")
        
        return {
            "status": "success",
            "session_id": session_id,
            "files": files,
            "file_count": len(files),
            "output_dir": output_dir,
            "output_contents": os.listdir(output_dir) if os.path.exists(output_dir) else []
        }
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in list_outputs: {e}\n{error_traceback}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "traceback": error_traceback
            }
        )

@app.get("/download/{path:path}")
async def download_file_any(path: str):
    """
    下载文件 - 支持多种URL格式：
    1. /download/{session_id}/{filename} (推荐)
    2. /download/{filename} (自动查找最新会话)
    """
    try:
        # 解析路径
        parts = path.split("/")
        
        # 情况1: /download/session_id/filename
        if len(parts) >= 2:
            session_id = parts[0]
            filename = "/".join(parts[1:])
            
            session_dir = os.path.join(TEMP_UPLOAD_DIR, session_id)
            if not os.path.exists(session_dir):
                return JSONResponse(
                    status_code=404,
                    content={
                        "status": "error",
                        "message": f"Session not found: {session_id}"
                    }
                )
        else:
            # 情况2: /download/filename (插件使用的格式)
            filename = path
            
            # 自动查找包含该文件的最新会话
            latest_session = None
            latest_file_path = None
            latest_time = 0
            
            if os.path.exists(TEMP_UPLOAD_DIR):
                for session_id in os.listdir(TEMP_UPLOAD_DIR):
                    session_dir = os.path.join(TEMP_UPLOAD_DIR, session_id)
                    if os.path.isdir(session_dir):
                        # 在多个可能的位置查找文件
                        possible_dirs = [
                            os.path.join(session_dir, "output"),
                            os.path.join(session_dir, "output", "extra_output"),
                            session_dir
                        ]
                        
                        for possible_dir in possible_dirs:
                            if os.path.exists(possible_dir):
                                file_path = os.path.join(possible_dir, filename)
                                if os.path.exists(file_path):
                                    # 记录文件修改时间
                                    file_time = os.path.getmtime(file_path)
                                    if file_time > latest_time:
                                        latest_time = file_time
                                        latest_session = session_id
                                        latest_file_path = file_path
                                        break
            
            if not latest_session or not latest_file_path:
                return JSONResponse(
                    status_code=404,
                    content={
                        "status": "error",
                        "message": f"File not found: {filename} in any session"
                    }
                )
            
            session_id = latest_session
            file_path = latest_file_path
            print(f"Auto-found file {filename} in session {session_id}")
            
            return FileResponse(
                path=file_path,
                filename=os.path.basename(filename),
                media_type="audio/wav"
            )
        
        # 在会话目录中查找文件
        possible_dirs = [
            os.path.join(session_dir, "output"),
            os.path.join(session_dir, "output", "extra_output"),
            session_dir
        ]
        
        file_path = None
        for possible_dir in possible_dirs:
            if os.path.exists(possible_dir):
                test_path = os.path.join(possible_dir, filename)
                if os.path.exists(test_path):
                    file_path = test_path
                    break
        
        if not file_path:
            # 搜索整个会话目录
            for root, dirs, files in os.walk(session_dir):
                if filename in files:
                    file_path = os.path.join(root, filename)
                    break
        
        if not file_path or not os.path.exists(file_path):
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"File not found: {filename} in session {session_id}",
                    "searched_dirs": possible_dirs
                }
            )
        
        return FileResponse(
            path=file_path,
            filename=os.path.basename(filename),
            media_type="audio/wav"
        )
        
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        )
    
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/infer/plugin")
async def infer_plugin(
    input_file: UploadFile = File(...),
    preset_path: str = Form(...),
    output_format: str = Form("wav"),
    extra_output_dir: bool = Form(False)
):
    """
    专门为插件设计的接口，返回完整的文件信息
    """
    try:
        # 创建临时会话目录
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(TEMP_UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # 验证预设路径
        if not os.path.exists(preset_path):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Preset file not found: {preset_path}"
                }
            )

        # 保存预设文件副本
        preset_copy_path = os.path.join(session_dir, os.path.basename(preset_path))
        shutil.copy(preset_path, preset_copy_path)

        # 保存音频文件
        input_dir = os.path.join(session_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        audio_file_path = os.path.join(input_dir, input_file.filename)
        with open(audio_file_path, "wb") as f:
            f.write(await input_file.read())

        # 输出目录
        output_dir = os.path.join(session_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # 调用 CLI 主逻辑
        from scripts.preset_infer_cli import main as cli_main
        cli_main(
            input_folder=input_dir,
            store_dir=output_dir,
            preset_path=preset_copy_path,
            output_format=output_format,
            extra_output_dir=extra_output_dir
        )

        # 直接返回文件列表，不需要插件再调用 /list_outputs
        target_dir = output_dir  # 插件使用 extra_output_dir=false
        
        # 查找人声文件（通常包含 "vocals" 或 "主唱" 等关键词）
        files = []
        vocals_file = None
        
        if os.path.exists(target_dir):
            for f in os.listdir(target_dir):
                if f.lower().endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(target_dir, f)
                    file_info = {
                        "name": f,
                        "path": file_path,
                        "url": f"http://127.0.0.1:9000/download/{session_id}/{f}",
                        "size": os.path.getsize(file_path)
                    }
                    files.append(file_info)
                    
                    # 识别是否为人声文件
                    if any(keyword in f.lower() for keyword in ["vocals", "vocal", "主唱", "人声", "voice"]):
                        vocals_file = file_info
        
        # 如果没有明确的人声文件，取第一个文件
        if not vocals_file and files:
            vocals_file = files[0]
        
        return {
            "status": "success", 
            "session_id": session_id,
            "output_dir": output_dir,
            "files": files,
            "vocals_file": vocals_file,
            "download_url": f"http://127.0.0.1:9000/download/{session_id}" if files else None
        }

    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500, 
            content={
                "status": "error", 
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        )

@app.get("/download/{session_id}")
async def download_session_files(session_id: str, filename: str = None):
    """
    下载会话中的文件
    """
    try:
        session_dir = os.path.join(TEMP_UPLOAD_DIR, session_id)
        if not os.path.exists(session_dir):
            return JSONResponse(
                status_code=404,
                content={"error": "Session not found"}
            )
        
        output_dir = os.path.join(session_dir, "output")
        if not os.path.exists(output_dir):
            return JSONResponse(
                status_code=404,
                content={"error": "Output directory not found"}
            )
        
        # 如果指定了文件名，下载单个文件
        if filename:
            file_path = os.path.join(output_dir, filename)
            if os.path.exists(file_path):
                return FileResponse(
                    path=file_path,
                    filename=filename,
                    media_type="audio/wav"
                )
            else:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"File {filename} not found"}
                )
        
        # 否则返回文件列表
        files = []
        for f in os.listdir(output_dir):
            if f.lower().endswith(('.wav', '.mp3', '.flac')):
                files.append({
                    "name": f,
                    "url": f"http://127.0.0.1:9000/download/{session_id}/{f}",
                    "size": os.path.getsize(os.path.join(output_dir, f))
                })
        
        return {
            "status": "success",
            "session_id": session_id,
            "files": files
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/preset_infer")
async def run_preset_infer(
    preset_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    output_format: str = Form("wav"),
    extra_output_dir: bool = Form(True),
    debug: bool = Form(False)
):
    """
    上传预设 JSON 和音频文件，返回处理后的音轨路径
    """
    try:
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(TEMP_UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # 保存预设
        preset_path = os.path.join(session_dir, preset_file.filename)
        with open(preset_path, "wb") as f:
            f.write(await preset_file.read())

        # 保存音频
        input_audio_dir = os.path.join(session_dir, "input")
        os.makedirs(input_audio_dir, exist_ok=True)
        audio_path = os.path.join(input_audio_dir, audio_file.filename)
        with open(audio_path, "wb") as f:
            f.write(await audio_file.read())

        # 输出目录
        output_dir = os.path.join(session_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        try:
            preset_infer(input_audio_dir, output_dir, preset_path, output_format, extra_output_dir)
            target_dir = os.path.join(output_dir, "extra_output") if extra_output_dir else output_dir
            if not os.path.exists(target_dir):
                target_dir = output_dir
            files = [os.path.join(target_dir, f) for f in os.listdir(target_dir)
                     if f.lower().endswith(('.wav', '.mp3', '.flac'))]
            return {"status": "success", "files": files}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_preset_api:app",
        host="127.0.0.1",
        port=9000,
        log_level="info"
    )
