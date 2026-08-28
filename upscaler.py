"""Real-ESRGAN による純粋な拡大（描き込みを足さない解像度アップ）。

Hires.fix（img2img で描き直す方式）は描き込みが増える代わりに絵柄が動く。
こちらは畳み込みネットワークを 1 回通すだけなので、元の絵をそのまま大きくする。
ステップの繰り返しも VAE も無いぶん速く、VRAM もほとんど使わない。

モデルはアニメ絵向けの RealESRGAN_x4plus_anime_6B（約 17MB）。初回だけ取得して
models/upscaler へ置き、以後は使い回す。
"""
from __future__ import annotations

import contextlib
import math
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

# 一度に処理する一辺の最大画素。実際にはプリセット解像度すべてがこれを超えるので、
# 常にタイルに分けて処理することになる。
#
# 256 にしている根拠 (RTX 5070 12GB で実測):
#   タイル推論の VRAM は一辺の二乗で増える。512 では 4.29GB 要るが、SDXL を
#   常駐させた状態の空きは 3.78GB しかなく、共有メモリへ溢れて 8.3 秒かかっていた。
#   256 なら 1.26GB に収まり 1.7 秒。枚数は増えるが総処理量はほぼ変わらないので、
#   溢れさえしなければ 512 と 256 で速度差はほとんど無い (1.3〜1.7 秒)。
#   羽根合わせを入れた今は、タイルサイズによる画質差も平均 0.19/255 以下。
TILE = 256

# 隣のタイルと重ねる幅。この帯の中で 2 枚の推論結果を混ぜて継ぎ目を消す。
# 64 にしている根拠:
#   - 帯が狭いと、混ぜても両側の推論結果の差が帯の中に凝縮されて残る。
#     このモデル (RRDB 6 block) は 1 画素の出力に広い範囲の入力を使うので、
#     タイルの端では本来見えるはずの周囲が欠けており、差はそれなりに出る
#   - 広げてもタイルが増えない。プリセット解像度 (832x1216 / 1024x1536 /
#     1536x1024 / 1024x1024) では 32 のときと枚数が同じで、
#     端に来る細いタイルの一辺はむしろ 64px -> 128px に太くなる
#   - VRAM のピークは 1 枚の大きさ (TILE) で決まるので、重ね幅を広げても増えない
TILE_OVERLAP = 64


class UpscalerError(RuntimeError):
    """モデルの取得や読み込みに失敗した（利用者に見せる想定のメッセージ）。"""


_model = None
_lock = threading.Lock()


def _say(on_phase, text: str) -> None:
    """いま何を待っているかを画面へ伝える。コールバックが無ければ何もしない。"""
    if on_phase is not None:
        on_phase(text)


def _ensure_weights(on_phase=None) -> Path:
    """モデルファイルを用意する。無ければ落とす。"""
    path = MODEL_DIR / MODEL_NAME
    if path.exists():
        return path
    # 初回だけネットワーク待ちが入る。黙って止まると回線の問題と区別が付かない
    _say(on_phase, "拡大用モデルをダウンロードしています... (初回のみ 約17MB)")
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


def _load(on_phase=None):
    """モデルを読み込む。初回だけ時間がかかり、以後は使い回す。"""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from spandrel import ModelLoader

        path = _ensure_weights(on_phase)
        _say(on_phase, "拡大用モデルを読み込んでいます...")
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


@contextlib.contextmanager
def _no_cudnn_benchmark():
    """拡大の間だけ cuDNN のカーネル探索を止める。

    engine 側が cudnn.benchmark を True にしてプロセス全体に効かせているため、
    何もしないとアップスケーラーもその下で動く。benchmark は「同じ形状を何度も
    回す」UNet では元が取れるが、ここでは形状ごとの総当たり探索に 17〜24 秒かかる
    割に、探索後の実行時間は 1.42 秒 対 1.40 秒 でほぼ変わらない (実測)。
    起動後の 1 回目だけ極端に待たされることになるので、ヒューリスティック選択で即決させる。
    engine._no_cudnn_benchmark と同じ対策だが、重い engine を読み込みたくないので
    ここに小さく持つ。
    """
    prev = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = prev


def _starts(total: int, tile: int, step: int) -> list[int]:
    """タイルの開始位置。最後の 1 枚は端に揃える。

    単純に range(0, total, step) で刻むと、末尾に幅の足りないタイルが残る。
    832px を TILE=256 / step=192 で刻むと最後は 64px になるが、その範囲は 1 つ前の
    タイル (576..832) が既に全幅で覆っている。周囲の情報が乏しいまま推論した結果を
    重ねるだけで、得るものが無い。開始位置を total - tile に寄せて常に全幅にする。
    """
    if total <= tile:
        return [0]
    xs = list(range(0, total - tile + 1, step))
    if xs[-1] != total - tile:
        xs.append(total - tile)
    return xs


def _ramp(n: int, device, dtype) -> torch.Tensor:
    """0 から 1 へ立ち上がる重み (Hann 窓の前半)。

    線形だと帯の両端で重みの傾きが折れ、その折れ目自体がうっすら筋として残る。
    コサインは両端で傾きが 0 になるので、外側の重み 1 の領域と滑らかに繋がる。
    """
    i = torch.arange(n, device=device, dtype=torch.float32)
    return (0.5 - 0.5 * torch.cos(math.pi * (i + 0.5) / n)).to(dtype)


def _tile_weight(h: int, w: int, pads, device, dtype) -> torch.Tensor:
    """タイル 1 枚分の重みマスク。

    pads は (上, 下, 左, 右) の立ち上げ幅。画像の縁に接する辺には 0 を渡す
    (縁には隣のタイルが無く、混ぜる相手がいないので 1 のままにする)。
    """
    wy = torch.ones(h, device=device, dtype=dtype)
    wx = torch.ones(w, device=device, dtype=dtype)
    t, b, l, r = pads
    # 細いタイルでは両側の帯が重なりうるので、重なった所は小さい方を採る
    if t:
        wy[:t] = torch.minimum(wy[:t], _ramp(t, device, dtype))
    if b:
        wy[h - b:] = torch.minimum(wy[h - b:], _ramp(b, device, dtype).flip(0))
    if l:
        wx[:l] = torch.minimum(wx[:l], _ramp(l, device, dtype))
    if r:
        wx[w - r:] = torch.minimum(wx[w - r:], _ramp(r, device, dtype).flip(0))
    return (wy[:, None] * wx[None, :])[None, None]


@torch.inference_mode()
def _run(model, x: torch.Tensor) -> torch.Tensor:
    """タイルに分けて通し、重なりを重み付きで混ぜて 1 枚に戻す。

    重なりを切り捨てて内側だけを貼ると、境界の左右で別々のタイルの推論結果が
    直に隣り合う。タイルの端は本来参照するはずの周囲が欠けたまま推論されるので
    中央部とは僅かに結果が違い、その差が 1 本の筋になって出る。
    そこで出力と重みの合計を別々に足し込み、最後に割って重み付き平均にする。
    重みは重なりの帯で 0 から 1 へ滑らかに変わるため、差は帯全体に引き伸ばされ、
    どこにも段差が立たない。
    """
    _, _, h, w = x.shape
    s = model.scale
    if h <= TILE and w <= TILE:
        return model(x)

    # 足し込みは fp32 で行う。fp16 のまま累積すると、重みで割った後の量子化誤差が
    # 平坦な面で階調の段になって出うる。832x1216 を 4 倍にする場合で 260MB ほど
    acc = torch.zeros((1, 3, h * s, w * s), device=x.device, dtype=torch.float32)
    wsum = torch.zeros((1, 1, h * s, w * s), device=x.device, dtype=torch.float32)
    step = TILE - TILE_OVERLAP
    for top in _starts(h, TILE, step):
        for left in _starts(w, TILE, step):
            bot, right = min(top + TILE, h), min(left + TILE, w)
            th, tw = bot - top, right - left
            tile = model(x[:, :, top:bot, left:right]).float()
            # 隣がある辺だけ立ち上げる。帯はタイルの半分を超えられない
            pads = [
                min(TILE_OVERLAP, th // 2) if top > 0 else 0,
                min(TILE_OVERLAP, th // 2) if bot < h else 0,
                min(TILE_OVERLAP, tw // 2) if left > 0 else 0,
                min(TILE_OVERLAP, tw // 2) if right < w else 0,
            ]
            m = _tile_weight(th * s, tw * s, [q * s for q in pads],
                             x.device, torch.float32)
            acc[:, :, top * s:bot * s, left * s:right * s] += tile * m
            wsum[:, :, top * s:bot * s, left * s:right * s] += m
    return (acc / wsum.clamp_min(1e-6)).to(x.dtype)


def upscale(img: Image.Image, scale: float = SCALE, on_phase=None) -> Image.Image:
    """画像を scale 倍にする。描き込みは足さず、元の絵を保ったまま拡大する。"""
    model = _load(on_phase)
    _say(on_phase, "拡大しています...")
    dev = next(model.model.parameters()).device
    dtype = next(model.model.parameters()).dtype

    with _no_cudnn_benchmark():
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
