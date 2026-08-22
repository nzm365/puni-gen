@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo .venv がありません。先に setup.bat を実行してください。
    pause
    exit /b 1
)
.venv\Scripts\python.exe app.py
pause
