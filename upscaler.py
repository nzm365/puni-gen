"""Real-ESRGAN による純粋な拡大（描き込みを足さない解像度アップ）。

Hires.fix（img2img で描き直す方式）は描き込みが増える代わりに絵柄が動く。
こちらは畳み込みネットワークを 1 回通すだけなので、元の絵をそのまま大きくする。
ステップの繰り返しも VAE も無いぶん速く、VRAM もほとんど使わない。

モデルはアニメ絵向けの RealESRGAN_x4plus_anime_6B（約 17MB）。初回だけ取得して
models/upscaler へ置き、以後は使い回す。
"""
from __future__ import annotations

import threading
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models" / "upscaler"
MODEL_NAME = "RealESRGAN_x4plus_anime_6B.pth"
MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
    "RealESRGAN_x4plus_anime_6B.pth"
)

# 拡大率。モデル自体は 4 倍なので、4 倍に上げてから目的の倍率へ縮める。
# 一度大きくしてから縮めた方が、直接の倍率で処理するより輪郭がなめらかになる
SCALE = 2.0

# 一度に処理する一辺の最大画素。これを超える画像はタイルに分けて処理する。
# 832x1216 程度なら分割不要だが、さらに大きい画像でも VRAM で詰まらないようにする
TILE = 512
TILE_OVERLAP = 32  # 継ぎ目が出ないよう重ねる幅


class UpscalerError(RuntimeError):
    """モデルの取得や読み込みに失敗した（利用者に見せる想定のメッセージ）。"""


_model = None
_lock = threading.Lock()


def _ensure_weights() -> Path:
    """モデルファイルを用意する。無ければ落とす。"""
    path = MODEL_DIR / MODEL_NAME
    if path.exists():
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    try:
        import requests

        with requests.get(MODEL_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with part.open("wb") as f:
                for chunk in r.iter_content(1024 * 256):
                    f.write(chunk)
    except Exception as e:  # noqa: BLE001
        if part.exists():
            part.unlink()
        raise UpscalerError(
            f"拡大用モデルの取得に失敗しました: {e}\n"
            "ネットワーク接続を確認してください（初回のみ約 17MB の取得が必要です）。"
        ) from e
    part.rename(path)
    return path


def _load():
    """モデルを読み込む。初回だけ時間がかかり、以後は使い回す。"""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from spandrel import ModelLoader

        path = _ensure_weights()
        try:
            model = ModelLoader().load_from_file(str(path))
        except Exception as e:  # noqa: BLE001
            raise UpscalerError(f"拡大用モデルを読み込めませんでした: {e}") from e
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _model = model.to(dev).eval()
        print(f"[upscaler] {MODEL_NAME} を読み込みました（{model.scale} 倍モデル）")
        return _model


def _to_tensor(img: Image.Image, device, dtype) -> torch.Tensor:
    import numpy as np

    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device, dtype)


def _to_image(t: torch.Tensor) -> Image.Image:
    import numpy as np

    a = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((a * 255.0).round().astype(np.uint8))


@torch.inference_mode()
def _run(model, x: torch.Tensor) -> torch.Tensor:
    """タイルに分けて通す。継ぎ目が出ないよう重ねて取り、中央だけを使う。"""
    _, _, h, w = x.shape
    s = model.scale
    if h <= TILE and w <= TILE:
        return model(x)

    out = torch.zeros((1, 3, h * s, w * s), device=x.device, dtype=x.dtype)
    step = TILE - TILE_OVERLAP
    for top in range(0, h, step):
        for left in range(0, w, step):
            bot, right = min(top + TILE, h), min(left + TILE, w)
            tile = model(x[:, :, top:bot, left:right])
            # 重ねた分は捨てて、内側だけを書き込む
            ct = TILE_OVERLAP // 2 if top > 0 else 0
            cl = TILE_OVERLAP // 2 if left > 0 else 0
            out[:, :, (top + ct) * s:bot * s, (left + cl) * s:right * s] = \
                tile[:, :, ct * s:, cl * s:]
    return out


def upscale(img: Image.Image, scale: float = SCALE) -> Image.Image:
    """画像を scale 倍にする。描き込みは足さず、元の絵を保ったまま拡大する。"""
    model = _load()
    dev = next(model.model.parameters()).device
    dtype = next(model.model.parameters()).dtype

    x = _to_tensor(img, dev, dtype)
    y = _run(model, x)
    big = _to_image(y)

    # モデルは 4 倍固定。目的の倍率へ落とす
    target = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    if big.size != target:
        big = big.resize(target, Image.LANCZOS)
    return big


def unload():
    """VRAM を空けたいときに手放す。"""
    global _model
    with _lock:
        _model = None
