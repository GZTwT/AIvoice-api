import io
import os
import sys
import logging

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
svc_root = os.environ.get("SVC_ROOT")
if svc_root:
    svc_root = os.path.abspath(os.path.join(_repo_root, svc_root))
    sys.path.insert(0, svc_root)

import soundfile
import torch
import torchaudio
from flask import Flask, request, send_file
from flask_cors import CORS

from inference.infer_tool import RealTimeVC, Svc

app = Flask(__name__)

CORS(app)

logging.getLogger('numba').setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # 变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # DAW所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    speaker_id = int(float(request_form.get("sSpeakId", 0)))
    # http获得wav文件并转换
    input_wav_path = io.BytesIO(wave_file.read())

        # 模型推理
    if raw_infer:
        # out_audio, out_sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path)
        out_audio, out_sr, _ = svc_model.infer(speaker_id, f_pitch_change, input_wav_path, cluster_infer_ratio=0,
                                               auto_predict_f0=False, noice_scale=0.4, f0_filter=False)
        tar_audio = torchaudio.functional.resample(out_audio, svc_model.target_sample, daw_sample)
    else:
        out_audio = svc.process(svc_model, speaker_id, f_pitch_change, input_wav_path, cluster_infer_ratio=0,
                                auto_predict_f0=False, noice_scale=0.4, f0_filter=False)
        tar_audio = torchaudio.functional.resample(torch.from_numpy(out_audio), svc_model.target_sample, daw_sample)
    # 返回音频
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio.cpu().numpy(), daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == '__main__':
    # 启用则为直接切片合成，False为交叉淡化方式
    # vst插件调整0.3-0.5s切片时间可以降低延迟，直接切片方法会有连接处爆音、交叉淡化会有轻微重叠声音
    # 自行选择能接受的方法，或将vst最大切片时间调整为1s，此处设为Ture，延迟大音质稳定一些
    raw_infer = True

    # ================ 模型与配置路径：优先环境变量，默认 logs/44k ================
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

    print(f"加载模型: {model_name}")
    print(f"配置文件: {config_name}")

    if not os.path.exists(model_name):
        print(f"错误: 模型文件不存在! {model_name}")
        sys.exit(1)
    if not os.path.exists(config_name):
        print(f"错误: 配置文件不存在! {config_name}")
        sys.exit(1)

    # ================ 应用 infer_tool 补丁（同 flask_api_full_song.py） ================
    # 修复 cluster_model_path=None 导致的路径拼接崩溃
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
                print(f"[OK] 已将 speech_encoder 从 '{self.hps_ms.model.speech_encoder}' 改为 'hubertsoft'")
                self.hps_ms.model.speech_encoder = 'hubertsoft'
                self.speech_encoder = 'hubertsoft'

    infer_tool_module.Svc.__init__ = fixed_svc_init
    print("[OK] 已应用 infer_tool 补丁")

    import importlib
    importlib.reload(infer_tool_module)
    from inference.infer_tool import Svc as SvcFixed, RealTimeVC as RealTimeVCFixed

    try:
        svc_model = SvcFixed(model_name, config_name,
                             device="cuda" if torch.cuda.is_available() else "cpu")
        svc = RealTimeVCFixed()
        print("[OK] 模型加载成功!")
        print(f"目标采样率: {svc_model.target_sample}")
        if hasattr(svc_model, 'spk2id') and svc_model.spk2id:
            print(f"可用的说话人: {list(svc_model.spk2id.keys())}")
    except Exception as e:
        print(f"[FAIL] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 此处与vst插件对应，不建议更改
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)
