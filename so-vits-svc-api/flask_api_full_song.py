import io
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
svc_root = os.environ.get("SVC_ROOT")
if svc_root:
    svc_root = os.path.abspath(os.path.join(_repo_root, svc_root))
    sys.path.insert(0, svc_root)

import numpy as np
import soundfile
from flask import Flask, request, send_file

from inference import infer_tool, slicer
from inference.infer_tool import F0FilterException

try:
    import torch
except ImportError:
    print("错误: 需要安装 torch 库")
    print("请运行: pip install torch torchaudio")
    sys.exit(1)

app = Flask(__name__)

# ================ infer_tool 补丁（启动即应用，避免双类副本坑） ================
import inference.infer_tool as infer_tool_module
original_init = infer_tool_module.Svc.__init__

def fixed_svc_init(self, net_g_path, config_path,
                   device=None,
                   cluster_model_path="",
                   nsf_hifigan_enhance=False,
                   diffusion_model_path="",
                   diffusion_config_path="configs/diffusion.yaml",
                   shallow_diffusion=False,
                   only_diffusion=False,
                   spk_mix_enable=False,
                   feature_retrieval=False):
    if cluster_model_path is None:
        cluster_model_path = ""
    if diffusion_model_path is None:
        diffusion_model_path = ""
    original_init(self, net_g_path, config_path,
                  device=device,
                  cluster_model_path=cluster_model_path,
                  nsf_hifigan_enhance=nsf_hifigan_enhance,
                  diffusion_model_path=diffusion_model_path,
                  diffusion_config_path=diffusion_config_path,
                  shallow_diffusion=shallow_diffusion,
                  only_diffusion=only_diffusion,
                  spk_mix_enable=spk_mix_enable,
                  feature_retrieval=feature_retrieval)
    if hasattr(self, 'hps_ms') and hasattr(self.hps_ms.model, 'speech_encoder'):
        if self.hps_ms.model.speech_encoder != 'hubertsoft':
            self.hps_ms.model.speech_encoder = 'hubertsoft'
            self.speech_encoder = 'hubertsoft'

infer_tool_module.Svc.__init__ = fixed_svc_init
import importlib
importlib.reload(infer_tool)
from inference import infer_tool as infer_tool_fixed

G_MODEL, G_CONFIG = None, None
MODEL_OPTS = {"device": "cpu", "enhance": False, "shallow_diffusion": False,
              "only_diffusion": False, "diffusion_model_path": "",
              "diffusion_config_path": "configs/diffusion.yaml",
              "feature_retrieval": False, "cluster_model_path": "",
              "use_spk_mix": False}


def split_list_by_n(list_collection, n, pre=0):
    for i in range(0, len(list_collection), n):
        yield list_collection[i - pre if i - pre >= 0 else i: i + n]


def build_svc(opts):
    global svc_model
    svc_model = infer_tool_fixed.Svc(
        G_MODEL, G_CONFIG,
        device=opts["device"],
        cluster_model_path=opts["cluster_model_path"],
        nsf_hifigan_enhance=opts["enhance"],
        diffusion_model_path=opts["diffusion_model_path"],
        diffusion_config_path=opts["diffusion_config_path"],
        shallow_diffusion=opts["shallow_diffusion"],
        only_diffusion=opts["only_diffusion"],
        feature_retrieval=opts["feature_retrieval"],
        spk_mix_enable=opts["use_spk_mix"])
    return svc_model

@app.route("/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": svc_model is not None,
        "available_speakers": list(svc_model.spk2id.keys()) if hasattr(svc_model, 'spk2id') else [],
        "sample_rate": svc_model.target_sample if svc_model else None,
        "device": MODEL_OPTS["device"],
        "enhance": MODEL_OPTS["enhance"],
        "shallow_diffusion": MODEL_OPTS["shallow_diffusion"],
        "only_diffusion": MODEL_OPTS["only_diffusion"]
    }, 200

@app.route("/", methods=["GET"])
def index():
    """主页，显示API使用方法"""
    return {
        "api_name": "So-VITS-SVC Flask API",
        "version": "1.0",
        "endpoints": {
            "/wav2wav": {
                "method": "POST",
                "description": "音频转换接口",
                "parameters": {
                    "audio_path": "音频文件路径",
                    "tran": "音调调整（整数）",
                    "spk": "说话人ID或名称",
                    "wav_format": "输出格式（默认wav）"
                }
            },
            "/health": {
                "method": "GET",
                "description": "健康检查"
            }
        },
        "available_speakers": list(svc_model.spk2id.keys()) if hasattr(svc_model, 'spk2id') else []
    }, 200

# @app.route("/wav2wav", methods=["POST"])
# def wav2wav():
#     request_form = request.form
#     audio_path = request_form.get("audio_path", None)  # wav文件地址
#     tran = int(float(request_form.get("tran", 0)))  # 音调
#     spk = request_form.get("spk", 0)  # 说话人(id或者name都可以,具体看你的config)
#     wav_format = request_form.get("wav_format", 'wav')  # 范围文件格式
#     infer_tool.format_wav(audio_path)
#     chunks = slicer.cut(audio_path, db_thresh=-40)
#     audio_data, audio_sr = slicer.chunks2audio(audio_path, chunks)
#     audio = []
#     for (slice_tag, data) in audio_data:
#         print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
#         length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
#         if slice_tag:
#             print('jump empty segment')
#             _audio = np.zeros(length)
#         else:
#             # padd
#             pad_len = int(audio_sr * 0.5)
#             data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
#             raw_path = io.BytesIO()
#             soundfile.write(raw_path, data, audio_sr, format="wav")
#             raw_path.seek(0)
#             out_audio, out_sr = svc_model.infer(spk, tran, raw_path)
#             svc_model.clear_empty()
#             _audio = out_audio.cpu().numpy()
#             pad_len = int(svc_model.target_sample * 0.5)
#             _audio = _audio[pad_len:-pad_len]
#         audio.extend(list(infer_tool.pad_array(_audio, length)))
#     out_wav_path = io.BytesIO()
#     soundfile.write(out_wav_path, audio, svc_model.target_sample, format=wav_format)
#     out_wav_path.seek(0)
#     return send_file(out_wav_path, download_name=f"temp.{wav_format}", as_attachment=True)

@app.route("/wav2wav", methods=["POST"])
def wav2wav():
    # 检查是否有文件上传
    if 'audio_file' not in request.files:
        return {"error": "No audio file provided"}, 400
    
    audio_file = request.files['audio_file']
    
    # 获取参数
    tran = int(float(request.form.get("tran", 0)))
    spk = request.form.get("spk", "xun2")
    if not isinstance(spk, int):
        spk = str(spk)
        if spk not in svc_model.spk2id:
            return {"error": f"说话人 '{spk}' 不在当前模型列表，可用: {list(svc_model.spk2id.keys())}"}, 400
    wav_format = request.form.get("format", "wav")
    slice_db = int(float(request.form.get("slice_db", -40)))
    f0_predictor = request.form.get("f0_predictor", "rmvpe")
    noise_scale = float(request.form.get("noise_scale", 0.4))
    cluster_infer_ratio = float(request.form.get("cluster_infer_ratio", 0))
    auto_predict_f0 = request.form.get("auto_predict_f0", "false").lower() == "true"
    f0_filter = request.form.get("f0_filter", "false").lower() == "true"
    cr_threshold = float(request.form.get("cr_threshold", 0.05))
    enhancer_adaptive_key = int(float(request.form.get("enhancer_adaptive_key", 0)))
    pad_seconds = float(request.form.get("pad_seconds", 0.5))
    clip_seconds = float(request.form.get("clip_seconds", 0))
    lg_num = float(request.form.get("lg_num", 0))
    lgr_num = float(request.form.get("lgr_num", 0.75))
    k_step = int(float(request.form.get("k_step", 100)))
    second_encoding = request.form.get("second_encoding", "false").lower() == "true"
    loudness_envelope_adjustment = float(request.form.get("loudness_envelope_adjustment", 1))

    # 创建临时文件
    import tempfile
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')

    try:
        # 保存上传的文件
        audio_file.save(temp_input.name)

        # 转换为 WAV 格式
        infer_tool.format_wav(temp_input.name)

        # 分割和处理音频
        chunks = slicer.cut(temp_input.name, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(temp_input.name, chunks)

        per_size = int(clip_seconds * audio_sr)
        lg_size = int(lg_num * audio_sr)
        lg_size_r = int(lg_size * lgr_num)
        lg_size_c_l = (lg_size - lg_size_r) // 2
        lg_size_c_r = lg_size - lg_size_r - lg_size_c_l
        lg = np.linspace(0, 1, lg_size_r) if lg_size != 0 else 0

        audio_parts = []
        for (slice_tag, data) in audio_data:
            print(f'Processing segment: {round(len(data) / audio_sr, 3)}s')

            length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
            if slice_tag:
                audio_parts.append(infer_tool.pad_array(np.zeros(length), length))
                continue
            if per_size != 0:
                datas = split_list_by_n(data, per_size, lg_size)
            else:
                datas = [data]
            for k, dat in enumerate(datas):
                per_length = int(np.ceil(len(dat) / audio_sr * svc_model.target_sample)) if clip_seconds != 0 else length
                pad_len = int(audio_sr * pad_seconds)
                dat = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
                raw_path = io.BytesIO()
                soundfile.write(raw_path, dat, audio_sr, format="wav")
                raw_path.seek(0)
                try:
                    out_audio, out_sr, _ = svc_model.infer(spk, tran, raw_path,
                                                           cluster_infer_ratio=cluster_infer_ratio,
                                                           auto_predict_f0=auto_predict_f0,
                                                           noice_scale=noise_scale,
                                                           f0_filter=f0_filter,
                                                           f0_predictor=f0_predictor,
                                                           enhancer_adaptive_key=enhancer_adaptive_key,
                                                           cr_threshold=cr_threshold,
                                                           k_step=k_step,
                                                           second_encoding=second_encoding,
                                                           loudness_envelope_adjustment=loudness_envelope_adjustment)
                # ponytail: 按名字匹配，规避 infer_tool 被重复加载成两个类副本的坑
                except Exception as exc:
                    if type(exc).__name__ == 'F0FilterException':
                        print(f'Skip segment: no voice detected ({round(len(dat) / audio_sr, 3)}s)')
                        _audio = np.zeros(per_length)
                    else:
                        raise
                else:
                    svc_model.clear_empty()
                    _audio = out_audio.cpu().numpy()
                    pad_len = int(svc_model.target_sample * pad_seconds)
                    _audio = _audio[pad_len:-pad_len]

                _audio = infer_tool.pad_array(_audio, per_length)
                if lg_size != 0 and k != 0 and audio_parts:
                    last = audio_parts[-1]
                    lg1 = last[-(lg_size_r + lg_size_c_r):-lg_size_c_r] if lgr_num != 1 else last[-lg_size:]
                    lg2 = _audio[lg_size_c_l:lg_size_c_l + lg_size_r] if lgr_num != 1 else _audio[0:lg_size]
                    audio_parts[-1] = last[0:-(lg_size_r + lg_size_c_r)] if lgr_num != 1 else last[0:-lg_size]
                    audio_parts.append(lg1 * (1 - lg) + lg2 * lg)
                    _audio = _audio[lg_size_c_l + lg_size_r:] if lgr_num != 1 else _audio[lg_size:]
                audio_parts.append(_audio)

        audio = np.concatenate(audio_parts) if audio_parts else np.zeros(0)

        # 保存输出
        soundfile.write(temp_output.name, audio, svc_model.target_sample, format=wav_format)

        # 返回文件
        return send_file(
            temp_output.name,
            download_name=f"output.{wav_format}",
            as_attachment=True
        )

    finally:
        # 清理临时文件
        try:
            os.unlink(temp_input.name)
            os.unlink(temp_output.name)
        except:
            pass


@app.route("/reload", methods=["POST"])
def reload_model():
    """动态重载模型：可切换 GPU/CPU、增强器、扩散等模型级参数"""
    global MODEL_OPTS
    if not request.form:
        return {"error": "缺少参数"}, 400
    opts = dict(MODEL_OPTS)
    device = request.form.get("device", "")
    if device:
        if device not in ("cuda", "cpu"):
            return {"error": f"device 只能是 cuda 或 cpu，收到: {device}"}, 400
        if device == "cuda" and not torch.cuda.is_available():
            return {"error": "当前环境没有可用的 CUDA 显卡"}, 400
        opts["device"] = device
    opts["enhance"] = request.form.get("enhance", "false").lower() == "true"
    opts["shallow_diffusion"] = request.form.get("shallow_diffusion", "false").lower() == "true"
    opts["only_diffusion"] = request.form.get("only_diffusion", "false").lower() == "true"
    opts["feature_retrieval"] = request.form.get("feature_retrieval", "false").lower() == "true"
    opts["use_spk_mix"] = request.form.get("use_spk_mix", "false").lower() == "true"
    dm = request.form.get("diffusion_model_path", "").strip()
    if dm:
        opts["diffusion_model_path"] = dm
    cm = request.form.get("cluster_model_path", "").strip()
    if cm:
        opts["cluster_model_path"] = cm

    try:
        build_svc(opts)
    except Exception as e:
        return {"error": f"模型重载失败: {e}"}, 500
    MODEL_OPTS = opts
    print(f"[OK] 模型重载完成: device={opts['device']} enhance={opts['enhance']} "
          f"shallow_diffusion={opts['shallow_diffusion']} only_diffusion={opts['only_diffusion']}")
    return {"status": "ok", "device": opts["device"], "enhance": opts["enhance"],
            "shallow_diffusion": opts["shallow_diffusion"], "only_diffusion": opts["only_diffusion"],
            "available_speakers": list(svc_model.spk2id.keys()) if hasattr(svc_model, 'spk2id') else []}, 200

if __name__ == '__main__':
    # ================ 模型路径配置 ================
    base = svc_root if svc_root else _repo_root
    model_name = os.environ.get("SVC_MODEL_PATH", "")
    if not model_name:
        model_name = os.path.join(base, "logs", "44k", "G_34400.pth")
    else:
        model_name = os.path.abspath(os.path.join(base, model_name))
    config_name = os.environ.get("SVC_CONFIG_PATH", "")
    if not config_name:
        config_name = os.path.join(base, "logs", "44k", "config.json")
    else:
        config_name = os.path.abspath(os.path.join(base, config_name))
    G_MODEL, G_CONFIG = model_name, config_name

    print(f"加载模型: {model_name}")
    print(f"配置文件: {config_name}")

    if not os.path.exists(model_name):
        print(f"错误: 模型文件不存在! {model_name}")
        sys.exit(1)
    if not os.path.exists(config_name):
        print(f"错误: 配置文件不存在! {config_name}")
        sys.exit(1)

    MODEL_OPTS["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        build_svc(MODEL_OPTS)
        print("[OK] 模型加载成功!")

        print(f"目标采样率: {svc_model.target_sample}")
        print(f"Hop size: {svc_model.hop_size}")
        if hasattr(svc_model, 'spk2id') and svc_model.spk2id:
            print(f"可用的说话人: {list(svc_model.spk2id.keys())}")
        else:
            print("未找到说话人信息，将使用默认说话人")

        print(f"设备: {MODEL_OPTS['device']}")
        print("启动 Flask 服务器在 http://127.0.0.1:1145")
        print("等待请求...")
        app.run(port=1145, host="127.0.0.1", debug=False, threaded=False)

    except Exception as e:
        print(f"[FAIL] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
