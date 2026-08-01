@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo   AI Voice 全流程处理 Web
echo ============================================
echo.
echo  启动中... Web 会自动拉起 MSST / SVC 服务
echo  （模型加载约 10~60 秒）
echo.
echo  启动完成后打开浏览器访问:
echo   http://127.0.0.1:8010
echo.
echo  关闭本窗口即可停止 Web（后端服务会随之结束）
echo ============================================
echo.
python app.py
pause
