@echo off
setlocal
cd /d "%~dp0"

echo [1/4] uv を確認中...
where uv >nul 2>nul
if errorlevel 1 (
    echo uv が見つかりません。ユーザー領域にインストールします（管理者権限不要）
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

echo [2/4] Python 3.12 をプロジェクト専用に取得...
uv python install 3.12

echo [3/4] 仮想環境 .venv を作成...
uv venv --python 3.12 .venv

echo [4/4] 依存パッケージをインストール（torch は CUDA 12.8 ビルド / RTX 40・50 系対応）...
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -r requirements.txt

echo.
echo セットアップ完了。GPU 認識を確認します:
.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available(), '/', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo.
echo 次は models\checkpoints にモデル、models\embeddings に埋め込みを置いて start.bat を実行してください。
pause
