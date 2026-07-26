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

try:
    import torch
except ImportError:
    print("错误: 需要安装 torch 库")
    print("请运行: pip install torch torchaudio")
    sys.exit(1)

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": svc_model is not None,
        "available_speakers": list(svc_model.spk2id.keys()) if hasattr(svc_model, 'spk2id') else [],
        "sample_rate": svc_model.target_sample if svc_model else None
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
    wav_format = request.form.get("format", "wav")
    
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
        chunks = slicer.cut(temp_input.name, db_thresh=-40)
        audio_data, audio_sr = slicer.chunks2audio(temp_input.name, chunks)
        
        audio = []
        for (slice_tag, data) in audio_data:
            print(f'Processing segment: {round(len(data) / audio_sr, 3)}s')
            
            length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
            if slice_tag:
                _audio = np.zeros(length)
            else:
                pad_len = int(audio_sr * 0.5)
                data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
                raw_path = io.BytesIO()
                soundfile.write(raw_path, data, audio_sr, format="wav")
                raw_path.seek(0)
                out_audio, out_sr = svc_model.infer(spk, tran, raw_path)
                svc_model.clear_empty()
                _audio = out_audio.cpu().numpy()
                pad_len = int(svc_model.target_sample * 0.5)
                _audio = _audio[pad_len:-pad_len]
            
            audio.extend(list(infer_tool.pad_array(_audio, length)))
        
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

if __name__ == '__main__':
    # ================ 修改位置1：定义模型和配置文件路径 ================
    import os
    base = svc_root if svc_root else _repo_root
    model_name = os.environ.get("SVC_MODEL_PATH", "")
    if not model_name:
        model_name = os.path.join(base, "logs", "44k", "G_55200.pth")
    else:
        model_name = os.path.abspath(os.path.join(base, model_name))
    config_name = os.environ.get("SVC_CONFIG_PATH", "")
    if not config_name:
        config_name = os.path.join(base, "logs", "44k", "config.json")
    else:
        config_name = os.path.abspath(os.path.join(base, config_name))
    
    print(f"加载模型: {model_name}")
    print(f"配置文件: {config_name}")
    
    # ================ 修改位置2：检查文件是否存在 ================
    if not os.path.exists(model_name):
        print(f"错误: 模型文件不存在! {model_name}")
        sys.exit(1)
    if not os.path.exists(config_name):
        print(f"错误: 配置文件不存在! {config_name}")
        sys.exit(1)
    
    # ================ 修改位置3：在导入后立即应用补丁 ================
    # 修复 infer_tool.py 中的 cluster_model_path 问题
    import inference.infer_tool as infer_tool_module
    
    # 保存原始的 Svc.__init__ 方法
    original_init = infer_tool_module.Svc.__init__
    
    # 创建修复后的初始化方法
    def fixed_svc_init(self, net_g_path, config_path,
                       device=None,
                       cluster_model_path="",  # 确保默认是空字符串而不是None
                       nsf_hifigan_enhance=False,
                       diffusion_model_path="",
                       diffusion_config_path="configs/diffusion.yaml",
                       shallow_diffusion=False,
                       only_diffusion=False,
                       spk_mix_enable=False,
                       feature_retrieval=False):
        
        # 确保路径参数是字符串
        if cluster_model_path is None:
            cluster_model_path = ""
        if diffusion_model_path is None:
            diffusion_model_path = ""
        
        # 调用原始初始化
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
        
        # 确保 speech_encoder 使用 hubertsoft（可选）
        if hasattr(self, 'hps_ms') and hasattr(self.hps_ms.model, 'speech_encoder'):
            if self.hps_ms.model.speech_encoder != 'hubertsoft':
                print(f"✓ 已将 speech_encoder 从 '{self.hps_ms.model.speech_encoder}' 改为 'hubertsoft'")
                self.hps_ms.model.speech_encoder = 'hubertsoft'
                self.speech_encoder = 'hubertsoft'
    
    # 替换初始化方法
    infer_tool_module.Svc.__init__ = fixed_svc_init
    print("✓ 已应用 infer_tool 补丁")
    
    # ================ 修改位置4：重新导入 infer_tool 以使用修复的版本 ================
    # 重新导入确保使用修复后的版本
    import importlib
    importlib.reload(infer_tool)
    from inference import infer_tool as infer_tool_fixed
    
    # ================ 修改位置5：加载模型 ================
    try:
        # 使用修复后的 infer_tool
        svc_model = infer_tool_fixed.Svc(
            model_name, 
            config_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("✓ 模型加载成功!")
        
        # ================ 修改位置6：打印模型信息 ================
        print(f"目标采样率: {svc_model.target_sample}")
        print(f"Hop size: {svc_model.hop_size}")
        if hasattr(svc_model, 'spk2id') and svc_model.spk2id:
            print(f"可用的说话人: {list(svc_model.spk2id.keys())}")
        else:
            print("未找到说话人信息，将使用默认说话人")
        
        # ================ 修改位置7：启动服务器 ================
        print("启动 Flask 服务器在 http://127.0.0.1:1145")
        print("等待请求...")
        app.run(port=1145, host="127.0.0.1", debug=False, threaded=False)
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)