import os
import json
import subprocess
import threading
import urllib.parse
import uuid
import webbrowser
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)

# 后端服务路径（可用环境变量覆盖）
MSST_WEBUI = os.environ.get("FLOW_MSST_ROOT", r"E:\github\aivoice\MSST-WebUI")
SVC_ROOT = os.environ.get("FLOW_SVC_ROOT", r"E:\github\aivoice\So-VITS-SVC\newpackage\so-vits-svc_tsss\so-vits-svc")
MSST_PY = os.path.join(MSST_WEBUI, "workenv", "python.exe")
SVC_PY = os.path.join(SVC_ROOT, "workenv", "python.exe")
MSST_SCRIPT = os.path.join(REPO, "msst-api", "fastapi_preset_api.py")
SVC_SCRIPT = os.path.join(REPO, "so-vits-svc-api", "flask_api_full_song.py")

MSST_URL = "http://127.0.0.1:9000"
SVC_URL = "http://127.0.0.1:1145"
WORK = os.path.join(BASE, "work")
os.makedirs(WORK, exist_ok=True)
SVC_PRESET_DIR = os.path.join(BASE, "svc_presets")
os.makedirs(SVC_PRESET_DIR, exist_ok=True)

SERVICES = {"msst": {"proc": None}, "svc": {"proc": None}}


@asynccontextmanager
async def lifespan(app):
    # Web 就绪后自动打开浏览器
    if os.environ.get("FLOW_NO_BROWSER") != "1":
        threading.Timer(6, lambda: webbrowser.open("http://127.0.0.1:8010")).start()
    # 启动 Web 时自动拉起两个后端服务（已运行则跳过）
    for name in ("msst", "svc"):
        try:
            if not healthy(MSST_URL if name == "msst" else SVC_URL):
                pid = start_service(name)
                print(f"[auto-start] {name} 已启动 pid={pid}")
        except Exception as e:
            print(f"[auto-start] {name} 启动失败: {e}")
    yield
    # Web 退出时停止自己拉起的后端服务
    for name in ("msst", "svc"):
        try:
            stop_service(name)
        except Exception as e:
            print(f"[auto-stop] {name} 停止失败: {e}")


app = FastAPI(title="AI Voice 全流程处理", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def healthy(url):
    try:
        return requests.get(f"{url}/health", timeout=3).status_code == 200
    except Exception:
        return False


CURRENT = {"svc_model": None, "svc_config": None}
TASKS = {}


def scan_svc_models():
    """扫描 SVC 项目内可用模型（logs/*/ 与 models/），返回模型+配置+说话人"""
    out, seen = [], set()

    def add(cfg_path):
        try:
            import json
            with open(cfg_path, encoding="utf-8") as f:
                spks = list(json.load(f).get("spk", {}).keys())
            d = os.path.dirname(cfg_path)
            gths = [f for f in os.listdir(d) if f.startswith("G_") and f.endswith(".pth")]
            if not gths or cfg_path in seen:
                return
            seen.add(cfg_path)
            gth = max(gths, key=lambda f: int(f.split("_")[1].split(".")[0]))
            name = "+".join(spks) if spks else os.path.basename(d)
            out.append({"name": f"{name} ({os.path.relpath(os.path.join(d, gth), SVC_ROOT)})",
                        "model": os.path.join(d, gth), "config": cfg_path, "speakers": spks})
        except Exception:
            pass

    logs = os.path.join(SVC_ROOT, "logs")
    if os.path.isdir(logs):
        for sr in os.listdir(logs):
            add(os.path.join(logs, sr, "config.json"))
    mdir = os.path.join(SVC_ROOT, "models")
    if os.path.isdir(mdir):
        for f in os.listdir(mdir):
            if f.endswith(".json"):
                add(os.path.join(mdir, f))
    return out


def start_service(name, extra_env=None):
    if name == "msst":
        cmd, cwd = [MSST_PY, MSST_SCRIPT], MSST_WEBUI
        env = os.environ.copy()
        env["MSST_ROOT"] = MSST_WEBUI
    else:
        cmd, cwd = [SVC_PY, SVC_SCRIPT], SVC_ROOT
        env = os.environ.copy()
        env["SVC_ROOT"] = SVC_ROOT
        env["PATH"] = os.path.join(SVC_ROOT, "ffmpeg", "bin") + os.pathsep + env.get("PATH", "")
        if extra_env is None and CURRENT.get("svc_model"):
            extra_env = {"SVC_MODEL_PATH": CURRENT["svc_model"], "SVC_CONFIG_PATH": CURRENT["svc_config"]}
    if extra_env:
        env.update(extra_env)
    logf = open(os.path.join(WORK, f"{name}.log"), "ab", buffering=0)
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=logf, stderr=logf,
                            stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    SERVICES[name]["proc"] = proc
    SERVICES[name]["log"] = logf
    return proc.pid


def find_pid(port):
    """按端口找监听进程 PID（兜底清理孤儿进程）"""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "TCP" and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
                return int(parts[4])
    except Exception:
        pass
    return None


def stop_service(name):
    port = 9000 if name == "msst" else 1145
    proc = SERVICES[name].get("proc")
    if proc and proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    pid = find_pid(port)
    if pid:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    SERVICES[name]["proc"] = None
    SERVICES[name].pop("log", None)


@app.post("/api/shutdown")
def shutdown():
    """停止全部服务并退出 Web，释放 8010 端口"""
    for name in ("msst", "svc"):
        try:
            stop_service(name)
        except Exception:
            pass
    threading.Thread(target=lambda: os._exit(0), daemon=True).start()
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE, "index.html"), "r", encoding="utf-8") as f:
        resp = HTMLResponse(f.read())
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/services")
def services():
    out = {}
    for name, s in SERVICES.items():
        proc = s.get("proc")
        running = proc is not None and proc.poll() is None
        out[name] = {"running": running,
                     "healthy": healthy(MSST_URL if name == "msst" else SVC_URL)}
    return out


@app.post("/api/services/{name}/start")
def start(name: str):
    if name not in SERVICES:
        return JSONResponse({"error": "unknown service"}, status_code=404)
    if healthy(MSST_URL if name == "msst" else SVC_URL):
        return {"status": "already running"}
    pid = start_service(name)
    return {"status": "starting", "pid": pid}


@app.post("/api/services/{name}/stop")
def stop(name: str):
    if name not in SERVICES:
        return JSONResponse({"error": "unknown service"}, status_code=404)
    stop_service(name)
    return {"status": "stopped"}


@app.get("/api/presets")
def presets():
    try:
        r = requests.get(f"{MSST_URL}/presets", timeout=5)
        return r.json() if r.status_code == 200 else {"presets": []}
    except Exception:
        return {"presets": [], "error": "MSST 未启动"}


@app.get("/api/svc_presets")
def svc_presets_list():
    out = []
    for f in os.listdir(SVC_PRESET_DIR):
        if f.endswith(".json"):
            out.append({"name": f[:-5]})
    return {"presets": sorted(out, key=lambda x: x["name"])}


@app.post("/api/svc_presets/save")
async def svc_preset_save(name: str = Form(...), params: str = Form(...)):
    name = os.path.basename(name.strip())
    if not name:
        return JSONResponse({"error": "预设名不能为空"}, status_code=400)
    try:
        data = json.loads(params)
    except Exception:
        return JSONResponse({"error": "参数不是合法 JSON"}, status_code=400)
    with open(os.path.join(SVC_PRESET_DIR, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"status": "ok", "name": name}


@app.post("/api/svc_presets/load")
def svc_preset_load(name: str = Form(...)):
    path = os.path.join(SVC_PRESET_DIR, os.path.basename(name) + ".json")
    if not os.path.exists(path):
        return JSONResponse({"error": f"预设 '{name}' 不存在"}, status_code=404)
    with open(path, encoding="utf-8") as f:
        return {"status": "ok", "params": json.load(f)}


@app.get("/api/models")
def models():
    return {"models": scan_svc_models()}


@app.post("/api/models/select")
def select_model(model_path: str = Form(...), config_path: str = Form(...)):
    CURRENT["svc_model"], CURRENT["svc_config"] = model_path, config_path
    if find_pid(1145):
        stop_service("svc")
        start_service("svc")
    return {"status": "ok"}


@app.get("/api/speakers")
def speakers():
    try:
        r = requests.get(f"{SVC_URL}/health", timeout=5)
        return {"speakers": r.json().get("available_speakers", [])}
    except Exception:
        return {"speakers": [], "error": "SVC 未启动"}


@app.get("/api/svc_state")
def svc_state():
    try:
        r = requests.get(f"{SVC_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else {"error": "SVC 未启动"}
    except Exception:
        return {"error": "SVC 未启动"}


@app.post("/api/svc_reload")
async def svc_reload(
    device: str = Form(""),
    enhance: str = Form("0"),
    shallow_diffusion: str = Form("0"),
    only_diffusion: str = Form("0"),
    feature_retrieval: str = Form("0"),
    diffusion_model_path: str = Form(""),
    cluster_model_path: str = Form(""),
):
    if not healthy(SVC_URL):
        return JSONResponse({"error": "SVC 服务未启动"}, status_code=500)
    data = {"enhance": "true" if enhance == "1" else "false",
            "shallow_diffusion": "true" if shallow_diffusion == "1" else "false",
            "only_diffusion": "true" if only_diffusion == "1" else "false",
            "feature_retrieval": "true" if feature_retrieval == "1" else "false"}
    if device:
        data["device"] = device
    if diffusion_model_path:
        data["diffusion_model_path"] = diffusion_model_path
    if cluster_model_path:
        data["cluster_model_path"] = cluster_model_path
    try:
        r = requests.post(f"{SVC_URL}/reload", data=data, timeout=300)
        return r.json() if r.status_code == 200 else JSONResponse(r.json(), status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"重载失败: {e}"}, status_code=500)
@app.post("/api/process")
async def process(
    audio: UploadFile = File(...),
    do_separate: str = Form("1"),
    preset: str = Form(""),
    spk: str = Form(""),
    tran: int = Form(0),
    fmt: str = Form("wav"),
    slice_db: int = Form(-40),
    f0_predictor: str = Form("rmvpe"),
    noise_scale: float = Form(0.4),
    cluster_infer_ratio: float = Form(0),
    auto_predict_f0: str = Form("0"),
    f0_filter: str = Form("0"),
    cr_threshold: float = Form(0.05),
    enhancer_adaptive_key: int = Form(0),
    pad_seconds: float = Form(0.5),
    clip_seconds: float = Form(0),
    lg_num: float = Form(0),
    lgr_num: float = Form(0.75),
    k_step: int = Form(100),
    second_encoding: str = Form("0"),
    loudness_envelope_adjustment: float = Form(1),
    start_time: float = Form(0),
    end_time: float = Form(0),
):
    if not healthy(SVC_URL):
        return JSONResponse({"error": "SVC 服务未启动，请先在上方启动"}, status_code=500)
    if do_separate == "1" and not healthy(MSST_URL):
        return JSONResponse({"error": "MSST 服务未启动，请先在上方启动"}, status_code=500)
    session_dir = os.path.join(WORK, uuid.uuid4().hex)
    os.makedirs(session_dir, exist_ok=True)
    ext = os.path.splitext(audio.filename or "")[1] or ".wav"
    src = os.path.join(session_dir, "input" + ext)
    with open(src, "wb") as f:
        f.write(await audio.read())

    task_id = uuid.uuid4().hex
    TASKS[task_id] = {"status": "running", "progress": 5, "stage": "预处理",
                      "session": os.path.basename(session_dir)}
    params = dict(do_separate=do_separate, preset=preset, spk=spk, tran=tran, fmt=fmt,
                  src=src, ext=ext, session_dir=session_dir, slice_db=slice_db,
                  f0_predictor=f0_predictor, noise_scale=noise_scale,
                  cluster_infer_ratio=cluster_infer_ratio, auto_predict_f0=auto_predict_f0,
                  f0_filter=f0_filter, cr_threshold=cr_threshold,
                  enhancer_adaptive_key=enhancer_adaptive_key, pad_seconds=pad_seconds,
                  clip_seconds=clip_seconds, lg_num=lg_num, lgr_num=lgr_num,
                  k_step=k_step, second_encoding=second_encoding,
                  loudness_envelope_adjustment=loudness_envelope_adjustment,
                  start_time=start_time, end_time=end_time)
    threading.Thread(target=run_process, args=(task_id, params), daemon=True).start()
    return {"task_id": task_id}


def run_process(task_id, p):
    t = TASKS[task_id]
    try:
        current, steps = p["src"], []
        if p.get("end_time", 0) > p.get("start_time", 0):
            t.update(progress=8, stage="裁剪音频区间")
            trimmed = os.path.join(p["session_dir"], "input_trim.wav")
            subprocess.run(["ffmpeg", "-y", "-i", p["src"],
                            "-ss", str(p["start_time"]), "-t", str(p["end_time"] - p["start_time"]),
                            "-c:a", "pcm_s16le", trimmed],
                           capture_output=True, check=True)
            p["src"] = trimmed
            p["ext"] = ".wav"
            steps.append({"step": "裁剪区间", "file": f"{p['start_time']}s - {p['end_time']}s"})
        if p["do_separate"] == "1":
            t.update(progress=10, stage="MSST 分离中")
            try:
                pl = requests.get(f"{MSST_URL}/presets", timeout=10).json().get("presets", [])
            except Exception:
                raise RuntimeError("MSST 预设列表获取失败")
            pf = next((x["filepath"] for x in pl if x["filename"] == p["preset"] or x["name"] == p["preset"]), None)
            if not pf:
                raise RuntimeError(f"预设 '{p['preset']}' 未找到")
            with open(p["src"], "rb") as f:
                r = requests.post(f"{MSST_URL}/infer/local",
                                  files={"audio_file": f},
                                  data={"preset_path": pf, "output_format": p["fmt"], "extra_output_dir": "true"},
                                  timeout=1800)
            rj = r.json()
            if r.status_code != 200 or not rj.get("files"):
                raise RuntimeError(f"MSST 分离失败: {json_dumps(rj)}")
            files = rj["files"]
            def _vbase(n):  # 小写文件名去扩展名
                return os.path.splitext(os.path.basename(n))[0].lower()
            _junk = ("other", "instrumental", "accompaniment", "drums", "bass", "music")
            cands = [x for x in files if _vbase(x["name"]).endswith("_vocals")] or \
                    [x for x in files if "vocal" in _vbase(x["name"]) or "人声" in x["name"]]
            cands = [x for x in cands if not any(k in _vbase(x["name"]) for k in _junk)]
            vocal = cands[0] if cands else files[0]
            vr = requests.get(f"{MSST_URL}/download/{rj['session_id']}/{urllib.parse.quote(vocal['name'])}", timeout=120)
            vocal_path = os.path.join(p["session_dir"], "vocal" + (os.path.splitext(vocal["name"])[1] or p["ext"]))
            with open(vocal_path, "wb") as f:
                f.write(vr.content)
            current = vocal_path
            steps.append({"step": "MSST 分离", "file": vocal["name"],
                          "download": f"/api/download/{t['session']}/{os.path.basename(vocal_path)}"})
            acc = next((x for x in files
                        if _vbase(x["name"]).endswith(("_other", "_instrumental", "_accompaniment", "_music"))
                        and "_vocals_" not in _vbase(x["name"])), None)
            if acc:
                ar = requests.get(f"{MSST_URL}/download/{rj['session_id']}/{urllib.parse.quote(acc['name'])}", timeout=120)
                acc_path = os.path.join(p["session_dir"], "accompaniment" + (os.path.splitext(acc["name"])[1] or p["ext"]))
                with open(acc_path, "wb") as f:
                    f.write(ar.content)
                steps.append({"step": "伴奏分离", "file": acc["name"],
                              "download": f"/api/download/{t['session']}/{os.path.basename(acc_path)}"})
            t.update(progress=60, stage="分离完成，准备变声")

        t.update(progress=65, stage="SVC 歌声转换中")
        try:
            cur_spks = requests.get(f"{SVC_URL}/health", timeout=5).json().get("available_speakers", [])
        except Exception:
            cur_spks = []
        if cur_spks and p["spk"] not in cur_spks:
            raise RuntimeError(f"说话人 '{p['spk']}' 与当前 SVC 模型不匹配（可用: {cur_spks}）。"
                               f"请刷新页面重新选择模型与说话人")
        data = {"spk": p["spk"], "tran": str(p["tran"]), "format": p["fmt"],
                "slice_db": str(p["slice_db"]), "f0_predictor": p["f0_predictor"],
                "noise_scale": str(p["noise_scale"]), "cluster_infer_ratio": str(p["cluster_infer_ratio"]),
                "auto_predict_f0": "true" if p["auto_predict_f0"] == "1" else "false",
                "f0_filter": "true" if p["f0_filter"] == "1" else "false",
                "cr_threshold": str(p["cr_threshold"]), "enhancer_adaptive_key": str(p["enhancer_adaptive_key"]),
                "pad_seconds": str(p["pad_seconds"]), "clip_seconds": str(p["clip_seconds"]),
                "lg_num": str(p["lg_num"]), "lgr_num": str(p["lgr_num"]),
                "k_step": str(p["k_step"]),
                "second_encoding": "true" if p["second_encoding"] == "1" else "false",
                "loudness_envelope_adjustment": str(p["loudness_envelope_adjustment"])}
        with open(current, "rb") as f:
            r = requests.post(f"{SVC_URL}/wav2wav", files={"audio_file": f}, data=data, timeout=1800)
        if r.status_code != 200:
            raise RuntimeError(f"SVC 转换失败: {r.text[:300]}")
        out = os.path.join(p["session_dir"], "output." + p["fmt"])
        with open(out, "wb") as f:
            f.write(r.content)
        steps.append({"step": "SVC 歌声转换", "file": os.path.basename(out)})
        t.update(progress=95, stage="转换完成")

        t.update(status="done", progress=100, stage="完成", steps=steps,
                 download=f"/api/download/{t['session']}/output.{p['fmt']}")
        if os.environ.get("FLOW_AUTO_OPEN") != "0":
            os.startfile(p["session_dir"])
    except Exception as e:
        t.update(status="error", progress=100, stage="失败", message=str(e))


@app.get("/api/task/{task_id}")
def task_status(task_id: str):
    return TASKS.get(task_id, {"status": "unknown"})


@app.get("/api/download/{session}/{filename}")
def download(session: str, filename: str):
    return FileResponse(os.path.join(WORK, session, os.path.basename(filename)))


@app.post("/api/mix")
async def mix(session: str = Form(...), vocal: str = Form("output.wav"),
              accomp: str = Form("accompaniment.wav"),
              vocal_vol: float = Form(1), accomp_vol: float = Form(1)):
    base = os.path.join(WORK, session)
    if not os.path.isdir(base):
        return JSONResponse({"error": "会话不存在"}, status_code=404)
    vocal, accomp = os.path.basename(vocal), os.path.basename(accomp)
    vp, ap = os.path.join(base, vocal), os.path.join(base, accomp)
    if not (os.path.isfile(vp) and os.path.isfile(ap)):
        return JSONResponse({"error": "音频文件不存在"}, status_code=404)
    out = os.path.join(base, "mix.wav")
    fc = (f"[0:a]volume={vocal_vol:.2f}[a0];"
          f"[1:a]volume={accomp_vol:.2f}[a1];"
          f"[a0][a1]amix=inputs=2:duration=longest:normalize=0")
    subprocess.run(["ffmpeg", "-y", "-i", vp, "-i", ap, "-filter_complex", fc,
                    "-c:a", "pcm_s16le", out], capture_output=True, check=True)
    return {"download": f"/api/download/{session}/mix.wav"}


def json_dumps(x):
    import json
    return json.dumps(x, ensure_ascii=False)[:500]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)
