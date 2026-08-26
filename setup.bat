@echo off
rem このファイルは CP932 (Shift_JIS) で保存すること。
rem UTF-8 + chcp 65001 だと、cmd が行の読み取り位置を多バイト文字の途中で
rem 見失い、日本語コメントの途中から命令として実行されてしまう
rem （実際に 'きの手順は' is not recognized... というエラーが出た）。
setlocal
set "PYTHONUTF8=1"
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
rem requirements.lock は推移的依存まで全てバージョンを固定してある。
rem これを使うことで、いつ・どの環境で実行しても同じ構成になる。
rem 依存を変えたいときの手順は README の「依存パッケージを更新する」を参照。
uv pip install --python .venv -r requirements.lock --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :failed
rem compel は notebook (Jupyter 一式) を依存に宣言しているが実行時には使わないため、
rem --no-deps で本体だけ入れる（必要な pyparsing はロック側で導入済み）
uv pip install --python .venv --no-deps compel==2.4.0
if errorlevel 1 goto :failed

echo.
echo セットアップ完了。GPU 認識を確認します:
.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available(), '/', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo.
echo 次は models\checkpoints にモデル、models\embeddings に埋め込みを置いて start.bat を実行してください。
pause
exit /b 0

:failed
echo.
echo インストールに失敗しました。ネットワーク接続を確認して、もう一度実行してください。
echo 繰り返し失敗する場合は .venv フォルダを削除してから再実行してください。
pause
exit /b 1
