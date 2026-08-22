# myimg

diffusers ベースの SDXL / Illustrious 系画像生成ツール（自分用・ComfyUI 非依存）。

## セットアップ
1. `setup.bat` を実行（uv で隔離環境を構築、既存の Python には触らない）
2. `models/checkpoints/` にチェックポイント、`models/embeddings/` に embedding を配置
3. `start.bat`

## モデル追加
`.safetensors` を置き、`presets/<ファイル名>.json` に推奨設定を書く（無ければ `_default.json`）。

## 注意
- RTX 50 系は torch の CUDA 12.8 ビルドが必須（setup.bat で対応済み）
- モデル・生成画像は git 管理外
