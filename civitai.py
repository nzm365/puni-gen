"""Civitai からチェックポイントを検索してダウンロードする。

ダウンロードには Civitai の API キーが必須（未認証だと 401）。キーは
config.local.json（.gitignore 済み）か環境変数 CIVITAI_TOKEN から読む。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / "models" / "checkpoints"
LORA_DIR = ROOT / "models" / "loras"
CONFIG = ROOT / "config.local.json"
THUMB_DIR = ROOT / ".thumb_cache"

API = "https://civitai.com/api/v1"

# ダウンロード URL は Civitai 生成のはずだが、API 応答が偽装された場合に
# API キー (Bearer) を無関係なホストへ送らないよう、送信先を限定する。
ALLOWED_DOWNLOAD_HOSTS = {"civitai.com", "www.civitai.com"}

# 落とすものの種類と、その保存先。LoRA も同じ検索・ダウンロードの仕組みに乗せる
KINDS = {"Checkpoint": CKPT_DIR, "LORA": LORA_DIR}


def _dir_for(kind: str) -> Path:
    """種類に対応する保存先。未知の種類はチェックポイント扱いにする。"""
    return KINDS.get(kind, CKPT_DIR)


def _safe_name(name: str, fallback: str) -> str:
    """API 由来の文字列を、保存先ディレクトリの外に出られないファイル名に落とす。

    ディレクトリ成分 (../ や C:\\ など) をすべて捨て、末尾要素だけを使う。
    空や '.'/'..' になったら fallback を返す。
    """
    base = Path(str(name).replace("\\", "/")).name
    if not base or base in (".", ".."):
        return fallback
    return base

# このツールは StableDiffusionXLPipeline 専用なので、SDXL 系以外は検索段階で除外する。
# SD1.5 や Flux を落としてもロードできない
SDXL_BASE_MODELS = [
    "SDXL 1.0",
    "SDXL 0.9",
    "SDXL Turbo",
    "Illustrious",
    "NoobAI",
    "Pony",
]

CHUNK = 1024 * 1024  # 1MB


class CivitaiError(RuntimeError):
    """検索・ダウンロードの失敗（利用者に見せる想定のメッセージを持つ）。"""


@dataclass
class Candidate:
    """検索結果 1 件（= 1 モデルバージョンの主ファイル）。"""

    model_name: str
    version_name: str
    base_model: str
    file_name: str
    size_bytes: int
    download_url: str
    thumb_path: str | None
    creator: str
    downloads: int
    # "Checkpoint" か "LORA"。保存先と、既にあるかの判定先が変わる
    kind: str = "Checkpoint"

    @property
    def label(self) -> str:
        return f"{self.model_name} [{self.version_name}]"

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1024**3

    @property
    def exists(self) -> bool:
        return (_dir_for(self.kind) / self.file_name).exists()


# ---------- API キー ----------
def load_token() -> str:
    if CONFIG.exists():
        try:
            token = json.loads(CONFIG.read_text(encoding="utf-8")).get("civitai_token", "")
            if token:
                return token
        except (json.JSONDecodeError, OSError):
            pass
    return os.environ.get("CIVITAI_TOKEN", "")


def save_token(token: str) -> str:
    data = {}
    if CONFIG.exists():
        try:
            data = json.loads(CONFIG.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    data["civitai_token"] = token.strip()
    CONFIG.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return "API キーを保存しました（config.local.json / git 管理外）。"


# ---------- 検索 ----------
def _pick_file(version: dict) -> dict | None:
    """バージョンの中から、落とすべき safetensors 本体を選ぶ。"""
    files = [
        f for f in version.get("files", [])
        if f.get("type") == "Model"
        and (f.get("metadata") or {}).get("format") == "SafeTensor"
        and f.get("downloadUrl")
    ]
    if not files:
        return None
    # primary 指定があればそれ。無ければ fp16 の pruned を優先（同じ中身で一番小さい）
    for f in files:
        if f.get("primary"):
            return f
    files.sort(key=lambda f: (
        (f.get("metadata") or {}).get("fp") != "fp16",
        (f.get("metadata") or {}).get("size") != "pruned",
    ))
    return files[0]


def placeholder() -> str:
    """サムネイルが取れなかったモデル用の灰色画像。"""
    THUMB_DIR.mkdir(exist_ok=True)
    path = THUMB_DIR / "_placeholder.png"
    if not path.exists():
        from PIL import Image

        Image.new("RGB", (320, 320), (60, 60, 66)).save(path)
    return str(path)


def _thumb(version: dict, key: str) -> str | None:
    """一番おとなしい画像をサムネイルとして取得する。失敗しても検索は続行。"""
    images = [i for i in version.get("images", []) if i.get("type") == "image" and i.get("url")]
    if not images:
        return None
    images.sort(key=lambda i: i.get("nsfwLevel") or 0)
    url = re.sub(r"/original=true/", "/width=320/", images[0]["url"])
    THUMB_DIR.mkdir(exist_ok=True)
    path = THUMB_DIR / f"{key}.jpg"
    if path.exists():
        return str(path)
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        path.write_bytes(r.content)
        return str(path)
    except (requests.RequestException, OSError):
        return None


def _get_models(params: list[tuple[str, str]], attempts: int = 4) -> list[dict]:
    """検索 API を叩く。Civitai は同じ問い合わせでも散発的に 503 を返すのでリトライする。"""
    last = ""
    for i in range(attempts):
        try:
            r = requests.get(f"{API}/models", params=params, timeout=30)
            if r.status_code >= 500:  # サーバ側の一時的な不調
                last = f"HTTP {r.status_code}"
                time.sleep(1.5 * (i + 1))
                continue
            r.raise_for_status()
            return r.json().get("items", [])
        except requests.RequestException as e:
            last = str(e)
            time.sleep(1.5 * (i + 1))
        except json.JSONDecodeError:
            raise CivitaiError("Civitai の応答を解釈できませんでした。") from None
    raise CivitaiError(
        f"Civitai に接続できませんでした（{last}）。"
        "時間をおいて、もう一度検索してみてください。"
    )


def search(query: str, limit: int = 20, kind: str = "Checkpoint") -> list[Candidate]:
    params = [("types", kind), ("limit", str(limit)), ("sort", "Most Downloaded")]
    if query.strip():
        params.append(("query", query.strip()))
    params += [("baseModels", b) for b in SDXL_BASE_MODELS]
    items = _get_models(params)

    out: list[Candidate] = []
    for item in items:
        for version in item.get("modelVersions", []):
            # モデルは複数バージョンを持ち、SDXL 以外が混ざることがある
            if version.get("baseModel") not in SDXL_BASE_MODELS:
                continue
            if version.get("paidAccess"):  # 購入・早期アクセス限定は落とせない
                continue
            f = _pick_file(version)
            if not f:
                continue
            # ファイル名は投稿者が付けた任意文字列。パス成分を除去し、
            # 拡張子も .safetensors に固定して保存先を checkpoints 内に閉じ込める
            fname = _safe_name(f.get("name", ""), f"model_{version.get('id')}.safetensors")
            if not fname.lower().endswith(".safetensors"):
                fname += ".safetensors"
            out.append(Candidate(
                model_name=item.get("name", "?"),
                version_name=version.get("name", "?"),
                base_model=version.get("baseModel", "?"),
                file_name=fname,
                size_bytes=int((f.get("sizeKB") or 0) * 1024),
                download_url=f["downloadUrl"],
                thumb_path=_thumb(version, _safe_name(str(version.get("id")), "thumb")),
                creator=(item.get("creator") or {}).get("username", "?"),
                downloads=(item.get("stats") or {}).get("downloadCount", 0),
                kind=kind,
            ))
            break  # 1 モデルにつき最新の SDXL 系バージョン 1 つだけ出す
    return out


# ---------- ダウンロード ----------
def download(cand: Candidate, on_progress=None) -> str:
    """models/checkpoints へ保存し、保存したファイル名を返す。

    中断しても .part に残り、次回は続きから再開する。完全に落ちきってから
    .safetensors に改名するので、中途半端なファイルがモデル一覧に出ることはない。
    """
    token = load_token()
    if not token:
        raise CivitaiError(
            "Civitai の API キーが未設定です。civitai.com にログインし、"
            "アカウント設定 → API Keys で作成したキーを上の欄に貼って保存してください。"
        )

    # 送信先ホストの確認（API キーを無関係なホストへ渡さない）
    host = urlparse(cand.download_url).hostname or ""
    if host.lower() not in ALLOWED_DOWNLOAD_HOSTS:
        raise CivitaiError(f"想定外のダウンロード先のため中止しました: {host or cand.download_url}")

    out_dir = _dir_for(cand.kind)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 保存名を再度サニタイズし、解決後のパスが checkpoints 配下に収まることを保証する
    safe = _safe_name(cand.file_name, "model.safetensors")
    dest = (out_dir / safe).resolve()
    if out_dir.resolve() not in dest.parents:
        raise CivitaiError("保存先が不正です（ファイル名を確認してください）。")
    if dest.exists():
        return f"すでに {safe} があります（ダウンロードは不要）。"

    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.exists() else 0

    free = shutil.disk_usage(out_dir).free
    if cand.size_bytes and free < cand.size_bytes - done + 1024**3:
        raise CivitaiError(
            f"ディスクの空きが足りません。必要 {cand.size_gb:.1f}GB に対し、"
            f"空きは {free / 1024**3:.1f}GB です。"
        )

    headers = {"Authorization": f"Bearer {token}"}
    if done:
        headers["Range"] = f"bytes={done}-"

    try:
        r = requests.get(cand.download_url, headers=headers, stream=True, timeout=60)
        if r.status_code == 401:
            raise CivitaiError(
                "API キーが拒否されました（401）。キーが正しいか、失効していないか確認してください。"
            )
        if r.status_code == 403:
            raise CivitaiError(
                "このモデルはダウンロードが許可されていません（403）。"
                "早期アクセスや購入が必要なモデルの可能性があります。"
            )
        if r.status_code == 416:  # 既に全部落ちている
            part.rename(dest)
            return f"{safe} を保存しました。"
        r.raise_for_status()

        if done and r.status_code != 206:  # 再開が拒否されたら最初から
            done = 0

        total = cand.size_bytes or (int(r.headers.get("Content-Length", 0)) + done)
        mode = "ab" if done else "wb"
        with part.open(mode) as fp:
            for chunk in r.iter_content(CHUNK):
                if not chunk:
                    continue
                fp.write(chunk)
                done += len(chunk)
                if on_progress and total:
                    on_progress(done / total, f"{done / 1024**3:.2f} / {total / 1024**3:.2f} GB")
    except requests.RequestException as e:
        raise CivitaiError(
            f"ダウンロードに失敗しました: {e}\n"
            "途中まで保存してあるので、もう一度押すと続きから再開します。"
        ) from e

    if cand.size_bytes and part.stat().st_size < cand.size_bytes:
        raise CivitaiError(
            "ダウンロードが途中で終わりました。もう一度押すと続きから再開します。"
        )

    part.rename(dest)
    where = "LoRA 一覧" if cand.kind == "LORA" else "モデル一覧"
    return f"{safe} を保存しました。{where}から選べます。"
