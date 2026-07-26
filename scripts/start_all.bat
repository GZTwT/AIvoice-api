@echo off
chcp 65001 >nul
echo ========================================
echo   AI Voice API - 启动所有服务
echo ========================================
echo.
echo 请确保已配置 .env 文件中的路径
echo.

REM 启动 So-VITS-SVC API (端口 1145)
echo [1/3] 启动 So-VITS-SVC 歌声转换 API ...
start "So-VITS-SVC" cmd /c "cd /d %~dp0..\so-vits-svc-api && python flask_api_full_song.py"

REM 启动 GPT-SoVITS API (端口 8000)
echo [2/3] 启动 GPT-SoVITS TTS API ...
start "GPT-SoVITS" cmd /c "cd /d %~dp0..\gpt-sovits-api && call start_cuda.bat"

REM 启动 MSST API (端口 1145)
echo [3/3] 启动 MSST 音频分离 API ...
start "MSST" cmd /c "cd /d %~dp0..\msst-api && uvicorn fastapi_preset_api:app --host 0.0.0.0 --port 1145"

echo.
echo 所有服务已启动！
pause
