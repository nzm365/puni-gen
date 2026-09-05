@echo off
rem このファイルは CP932 (Shift_JIS) で保存すること。
rem UTF-8 + chcp 65001 だと、cmd が行の読み取り位置を多バイト文字の途中で
rem 見失い、日本語コメントの途中から命令として実行されてしまう
rem （実際に 'きの手順は' is not recognized... というエラーが出た）。
rem
rem 以前は setup.bat と start.bat に分けていたが、分ける利点が無かったので 1 つにした。
rem そのぶん、何をしているかは [n/4] で毎回出す。
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
set "FIRST="
set "SKIPDEPS="

rem 起動したことが一目で分かるように名乗る。( ) \ / は echo にそのまま置ける
rem （cmd が特別扱いするのは % ^ & | < > で、この絵には含まれない）。
echo  ____  _  _  __ _  __  ___  ____  __ _ 
echo (  _ \/ )( \(  ( \(  )/ __)(  __)(  ( \
echo  ) __/) \/ (/    / )(( (_ \ ) _) /    /
echo (__)  \____/\_)__)(__)\___/(____)\_)__)
echo.
echo   Puni Uses No Intricacy.
echo.

echo [1/4] uv を確認...
where uv >nul 2>nul
if not errorlevel 1 goto :uv_ok
echo       見つかりません。ユーザー領域にインストールします（管理者権限は要りません）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
rem 入れた直後は PATH に反映されていないので、この実行の間だけ足す
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 goto :uv_missing
:uv_ok
echo       OK

:venv
echo [2/4] 実行環境を確認...
if exist .venv\Scripts\python.exe goto :venv_ok
echo       Python 3.12 をこのフォルダ専用に取得します（初回だけ。数分かかります）
uv python install 3.12
if errorlevel 1 goto :failed
echo       仮想環境 .venv を作ります
uv venv --python 3.12 .venv
if errorlevel 1 goto :failed
set "FIRST=1"
goto :deps
:venv_ok
echo       準備済み

:deps
echo [3/4] 依存パッケージを確認...
if defined SKIPDEPS goto :deps_skipped
rem requirements.lock は推移的依存まで全てバージョンを固定してある。
rem これを使うことで、いつ・どの環境で実行しても同じ構成になる。
rem 毎回走らせているのは、ロックが変わったときに自動で追いつくため。
rem 全部そろっていれば 18ms で終わり、ネットにも出ない（実測）。
uv pip install --python .venv -r requirements.lock --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :failed
rem compel は notebook (Jupyter 一式) を依存に宣言しているが実行時には使わないため、
rem --no-deps で本体だけ入れる（必要な pyparsing はロック側で導入済み）
uv pip install --python .venv --no-deps compel==2.4.0
if errorlevel 1 goto :failed
goto :gpu
:deps_skipped
echo       uv が無いので飛ばします。足りないものがあると、この後の起動で失敗します

:gpu
if not defined FIRST goto :run
echo.
echo       GPU の認識を確かめます（初回だけ）
.venv\Scripts\python.exe -c "import torch; print('      CUDA:', torch.cuda.is_available(), '/', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

:run
echo.
echo [4/4] PuniGen を起動します。ブラウザが自動で開きます...
echo       終わるときは、ブラウザのタブではなく、この黒い画面を閉じてください。
echo.
.venv\Scripts\python.exe app.py
pause
exit /b 0

:uv_missing
rem uv は無いが .venv があるなら、確認だけ飛ばして起動は試す
if not exist .venv\Scripts\python.exe goto :failed_uv
set "SKIPDEPS=1"
goto :venv

:failed_uv
echo.
echo uv を用意できませんでした。ネットワーク接続を確認して、もう一度実行してください。
pause
exit /b 1

:failed
echo.
echo 準備に失敗しました。ネットワーク接続を確認して、もう一度実行してください。
echo 繰り返し失敗する場合は .venv フォルダを削除してから、もう一度実行してください。
pause
exit /b 1
