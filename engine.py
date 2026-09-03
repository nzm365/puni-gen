"""diffusers ベースの SDXL/Illustrious 推論エンジン。ComfyUI 非依存。"""
from __future__ import annotations

import contextlib
import gc
import json
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

ROOT = Path(__file__).parent
import lora

CKPT_DIR = ROOT / "models" / "checkpoints"
EMB_DIR = ROOT / "models" / "embeddings"
PRESET_DIR = ROOT / "presets"
# fp16 VAE が NaN を出したモデル名の記録 (次回起動から最初から fp32 でデコードする)
VAE_FP32_FILE = ROOT / ".vae_fp32.json"

# 重み以外に要る作業用 VRAM のおおよその見積り（活性値・アテンション・VAE デコード）。
# 配置の判定はこの値を起点に決まるので、環境に合わせて調整するならここ。
ACTIVATION_HEADROOM_GB = 2.5

# オフロード時の作業領域。タイル分割デコード前提なので常駐時より小さくてよい。
# 8GB カードは OS が VRAM を掴んだ状態で空きが 7GB 前後になるため、
# ここを常駐と同じ 2.5 にすると UNet 4.8GB + 2.5GB = 7.3GB でロード自体が弾かれてしまう
OFFLOAD_HEADROOM_GB = 1.5

# 重みサイズを見積れなかったときのフォールバック閾値。
# 一般的な SDXL (fp16 で約 6.5GB) が常駐できるかどうかの目安。
LOW_VRAM_GB = 10

# ---------- 速度設定 ----------
# 旧 SPEED_TWEAKS は 4 つの施策を 1 つの定数に束ねていたため、fuse_qkv_projections が
# 重みを +約1.2GB して VRAM を溢れさせた事故 (RTX 5070 12GB で 0.92秒/step → 51秒/step、
# 2026-08-23 実測) の巻き添えで、メモリを一切増やさない TF32 / channels_last まで
# 無効化されていた。フラグ単位に分解し、それぞれ独立に測れるようにする。
USE_TF32 = True             # fp32 演算 (VAE の fp32 フォールバック時など) を高速化。メモリ増なし
USE_CHANNELS_LAST = True    # Tensor Core が効きやすいメモリ配置。メモリ増ほぼなし
USE_CUDNN_BENCHMARK = True  # 解像度が数種類に固定される本ツールと好相性。初回のみカーネル探索
USE_FUSE_QKV = False        # +約1.2GB。12GB では VRAM が溢れて逆効果 (上記実測)。16GB 以上専用

# ---------- 高解像度化 (Hires.fix) ----------
# 生成済み画像を拡大してから img2img で描き直し、細部を足す。
HIRES_SCALE = 1.5          # 倍率。832x1216 -> 1248x1824 (画素数 2.25 倍)
HIRES_STRENGTH = 0.35      # 変化の強さ。高すぎると元の構図が崩れる
HIRES_STEPS = 20           # 実効 step は steps x strength なので 20x0.35 = 7 step 相当

# 拡大後の UNet は画素数に比例して活性値が増える。12GB では 2 倍 (画素数 4 倍) が
# 載らないため既定は 1.5 倍 (画素数 2.25 倍)。それでも通常生成より重いので、
# この画素数を超えたらアテンションを分割して山を削る。
# 速度は落ちるが、共有メモリへ溢れて数十倍遅くなるよりはるかにまし。
# 1248x1824 = 2.28MP なので 1.5 倍でも分割が入る。
ATTENTION_SLICING_PIXELS = 2000 * 1000

# ---------- 変分 (variation seed) ----------
# 気に入った絵の seed を保ったまま、少しだけ違う絵を出す。A1111 の variation seed 相当。
# 元のノイズと別 seed のノイズを球面補間で混ぜる。0 で完全に同じ絵、1 で無関係な絵。
VARIATION_STRENGTH = 0.15  # 「もう少し違うのが欲しい」に合う程度の振れ幅
VARIATION_COUNT = 2        # 一度に出す枚数

# 単一ファイル形式のチェックポイントで UNet が持つキーの接頭辞。
# オフロード時は「一番大きい単体モジュール = UNet」が VRAM ピークになる。
UNET_PREFIX = "model.diffusion_model."


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995):
    """2 つのノイズを球面線形補間で混ぜる。

    単純な線形補間だと混ぜた分だけノルムが縮み、ノイズの分散が変わって
    絵が眠くなる。球面補間なら大きさを保ったまま向きだけを寄せられる。
    """
    v0f, v1f = v0.double(), v1.double()
    n0 = v0f / torch.norm(v0f)
    n1 = v1f / torch.norm(v1f)
    dot = (n0 * n1).sum()
    if dot.abs() > dot_threshold:  # ほぼ平行なら補間の意味がないので線形で十分
        out = (1 - t) * v0f + t * v1f
    else:
        theta0 = torch.acos(dot)
        sin0 = torch.sin(theta0)
        s0 = torch.sin((1 - t) * theta0) / sin0
        s1 = torch.sin(t * theta0) / sin0
        out = s0 * v0f + s1 * v1f
    return out.to(v0.dtype)


def safe_vae_dtype() -> torch.dtype:
    """fp16 VAE が NaN を出すモデルの退避先 dtype を返す。

    bf16 は fp32 と同じ値域を持ちながら fp16 並みに速いので第一候補。
    ただし Ampere (compute capability 8.0) 以上でないと対応しないため、
    対応していない GPU では fp32 に落とす（対象の RTX 30/40/50 は全て対応済み。
    範囲外の GPU で動かされたときに、静かに壊れず遅いだけで済むようにする保険）。
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _smi_free_bytes() -> int | None:
    """nvidia-smi が報告する空き VRAM。取れなければ None。

    Windows (WDDM) では torch.cuda.mem_get_info() が他プロセスの確保を反映しない。
    自分で確保した分は減るが、他のアプリが掴んでいる分は見えず、
    「OS が他を追い出せば自分が取れる量」に近い値が返る。実測:

        他プロセスが 3GB 保持   nvidia-smi 6.7GB / torch 10.8GB（保持前と同じ）

    その値を信じて常駐で載せると、実際には物理 VRAM に収まらず共有メモリへ
    溢れ、生成が 0.62s/step から 16.16s/step まで落ちた（同一 seed で 25 倍）。
    ドライバ同梱の nvidia-smi は物理的な空きを返すので、こちらも見て小さい方を採る。

    GPU が複数ある場合は、どの行が torch の見ている GPU なのかを
    CUDA_VISIBLE_DEVICES 次第で取り違えうるため、諦めて None を返す
    （従来どおり torch の値だけで判定する。今より悪くはならない）。
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            # Windows で一瞬コンソールが開くのを防ぐ
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None  # nvidia-smi が無い / 応答しない
    if r.returncode != 0:
        return None
    lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    if len(lines) != 1:  # 0 台 or 複数台。取り違えるより使わない
        return None
    try:
        return int(lines[0]) * 1024 * 1024
    except ValueError:
        return None


def free_vram_bytes() -> int:
    """いま実際に使える VRAM。torch と nvidia-smi の小さい方を採る。

    小さい方にするのは、多く見積もって載せてしまう方が害が大きいため。
    載らないものを弾きすぎても「他のアプリを閉じてください」と案内が出るだけだが、
    載せてしまうと共有メモリへ溢れて、原因の分からない激遅状態になる。
    """
    free = torch.cuda.mem_get_info()[0]
    smi = _smi_free_bytes()
    return free if smi is None else min(free, smi)


class InsufficientVram(RuntimeError):
    """VRAM が足りないと分かったのでロードを中止した。"""


def _say(on_phase, text: str) -> None:
    """いま何を待っているかを画面へ伝える。

    print はコンソール向けの詳細として残し、こちらには初心者が読んで
    「止まっていない」と分かる粒度の文だけを流す。コールバックが無ければ何もしない。
    """
    if on_phase is not None:
        on_phase(text)


def fp16_footprint(path: Path) -> tuple[float, float] | None:
    """safetensors のヘッダだけ読み、fp16 換算の (全体, UNet) を GB で返す。

    ヘッダを解釈できない形式なら None（＝見積りを諦めて従来どおりロードを試す）。
    """
    try:
        with path.open("rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if not 0 < n < 100 * 1024**2:  # 壊れたファイルで巨大な read をしない
                return None
            header = json.loads(f.read(n))
        total = unet = 0
        for key, meta in header.items():
            if key == "__metadata__" or "model_ema" in key:  # EMA は diffusers が捨てる
                continue
            numel = 1
            for d in meta["shape"]:
                numel *= d
            total += numel
            if key.startswith(UNET_PREFIX):
                unet += numel
    except Exception:  # noqa: BLE001
        return None
    if total == 0:
        return None
    per_elem_gb = 2 / 1024**3  # fp16 は 1 要素 2 byte
    return total * per_elem_gb, unet * per_elem_gb

SAMPLERS = {
    "euler": lambda cfg: EulerDiscreteScheduler.from_config(cfg),
    "euler_a": lambda cfg: EulerAncestralDiscreteScheduler.from_config(cfg),
    "dpmpp_2m": lambda cfg: DPMSolverMultistepScheduler.from_config(cfg, algorithm_type="dpmsolver++"),
    "dpmpp_2m_karras": lambda cfg: DPMSolverMultistepScheduler.from_config(
        cfg, algorithm_type="dpmsolver++", use_karras_sigmas=True
    ),
}


def list_checkpoints() -> list[str]:
    return sorted(p.name for p in CKPT_DIR.glob("*.safetensors"))


def fit_size(wh: tuple[int, int], budget_pixels: int) -> tuple[int, int]:
    """アスペクト比を保ったまま指定画素数に合わせ、8 の倍数に丸める (img2img 用)。"""
    w, h = wh
    s = (budget_pixels / (w * h)) ** 0.5
    return max(8, round(w * s / 8) * 8), max(8, round(h * s / 8) * 8)


def load_preset(ckpt_name: str) -> dict:
    """モデル名と同名の JSON があればそれを、なければ _default.json を返す。"""
    base = json.loads((PRESET_DIR / "_default.json").read_text(encoding="utf-8"))
    p = PRESET_DIR / f"{Path(ckpt_name).stem}.json"
    if p.exists():
        base.update(json.loads(p.read_text(encoding="utf-8")))
    return base


class Engine:
    def __init__(self):
        self.pipe: StableDiffusionXLPipeline | None = None
        self.pipe_i2i: StableDiffusionXLImg2ImgPipeline | None = None
        self.current: str | None = None
        self.loaded_embeddings: list[str] = []
        # 生成の中断フラグ。UI の中止ボタン → 次の step でループを打ち切る
        self.cancel = threading.Event()
        # ロードと生成の直列化。Gradio の demo.load はブラウザのタブごとに発火するため、
        # タブが複数あると同じモデルのロードが同時に走り、GPU に 2 個分載って溢れる。
        self._lock = threading.Lock()
        # いま _lock を握っている処理の名前。待たされた側が「何を待っているのか」を
        # 名指しできるようにするためだけに使う（ロックの正しさには関与しない）
        self._busy: str | None = None
        # いま載っている LoRA の (名前, 強度)。順序は set_adapters に渡す順と同じ
        self._loras: list[tuple[str, float]] = []
        self._offload = False
        # (モデル, プロンプト, ネガティブ, clip_skip) → encode_prompt の結果。
        # seed だけ変えて連打する使い方ではテキストエンコーダ 2 基を毎回回す必要がない
        self._embed_cache: OrderedDict[tuple, tuple] = OrderedDict()
        # 重み付き構文と長いプロンプトの解釈器 (CompelForSDXL)。初回使用時に作る
        self._compel = None
        # fp16 デコードで NaN/Inf を出した実績のあるモデル名 (以後そのモデルは fp32 でデコード)。
        # ディスクに永続化し、次回起動では fp16 で探り直さない（無駄な二重デコードを防ぐ）
        self._vae_fp32: set[str] = set()
        try:
            self._vae_fp32 = set(json.loads(VAE_FP32_FILE.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            pass
        # 見積りが外れて実際に載らなかった / bf16 でも NaN が出た、のように
        # 「予測ではなく実測で」タイル分割が要ると分かったときだけ立てる。
        # 立っている間は毎回の判定を飛ばし、モデルを入れ替えるまで戻さない
        self._vae_force_tiling = False
        if torch.cuda.is_available():
            if USE_CUDNN_BENCHMARK:
                torch.backends.cudnn.benchmark = True
            if USE_TF32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def _wait_note(self) -> str:
        """ロック待ちの文言。何を待っているのかを名指しする。

        ただ「他の処理」とだけ書くと、待たされている側からは何が起きているのか
        分からない。特にウォームアップは画面に出ない裏の処理なので、
        名前を出さないと理由の分からない待ちになる。
        """
        return f"{self._busy or '別の処理'}が終わるのを待っています..."

    # ---------- モデル ----------
    def load(self, ckpt_name: str, on_phase=None, loras=()) -> str:
        # ロックが空いていないときだけ待機中と伝える。生成中にモデルを切り替えると
        # ここで数十秒待つが、黙って止まって見えると固まったと誤解される
        if not self._lock.acquire(blocking=False):
            _say(on_phase, self._wait_note())
            self._lock.acquire()
        self._busy = "モデルの読み込み"
        try:
            return self._load_impl(ckpt_name, on_phase, loras)
        finally:
            self._busy = None
            self._lock.release()

    def _load_impl(self, ckpt_name: str, on_phase=None, loras=(), force=False) -> str:
        if not force and self.current == ckpt_name and self.pipe is not None:
            return f"already loaded: {ckpt_name}"
        self.unload()
        path = CKPT_DIR / ckpt_name
        _say(on_phase, "空き VRAM を確認しています...")
        offload, mode = self._plan(path, lora.total_bytes(n for n, _ in loras))
        # ここが全体で最も長い。6GB 級のファイルを読むので数十秒かかる
        gb = path.stat().st_size / 1024**3
        _say(on_phase, f"重みを読み込んでいます... ({gb:.1f}GB)")
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(path), torch_dtype=torch.float16, use_safetensors=True
        )
        self.pipe = pipe
        _say(on_phase, "embedding を読み込んでいます...")
        self._load_embeddings()  # オフロードのフックが付く前に済ませる
        self._load_lora_adapters(loras, on_phase)
        if USE_FUSE_QKV and not self._loras:
            # Q/K/V を 1 つの行列積に融合する (+1.2GB)。LoRA は to_q / to_k / to_v に
            # 別々に層を足すため、融合してしまうと当てる先が消える。LoRA を使うときは
            # 融合しない（既定は False なので、16GB 以上で有効にした人だけが影響を受ける）
            pipe.fuse_qkv_projections()
        # i2i パイプは t2i と全コンポーネントを共有する（重みのコピーは発生しない）。
        # オフロードのフックが付く前に作っておく。
        # torch_dtype は必ず明示する。diffusers 0.40 の from_pipe は dtype 未指定だと
        # 共有コンポーネントごと fp32 へキャストするため、重みが倍の 13GB になり
        # VRAM から溢れて生成が数十倍遅くなる（実測済み）。
        self.pipe_i2i = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe, torch_dtype=torch.float16)
        if offload:
            _say(on_phase, "CPU オフロードを準備しています... (VRAM が少ないため低速モード)")
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()  # 8GB 級では VAE デコードのピークも惜しい
        else:
            _say(on_phase, "GPU へ転送しています...")
            pipe.to("cuda")
            if USE_CHANNELS_LAST:
                # Tensor Core が効きやすいメモリ配置（オフロード時は転送が支配的なので省略）
                pipe.unet.to(memory_format=torch.channels_last)
                pipe.vae.to(memory_format=torch.channels_last)
            # 常駐時はタイル分割しない。プリセット解像度 (832x1216 / 1024x1536) は
            # 旧実装ではどちらも常にタイル分割を踏んでいた（しきい値が 1024px のため）。
            # 一括デコードの方が速く、タイル境界の継ぎ目も原理的に消える。
            # 万一 VRAM に載らなければ _decode_latents がデコードごとに判定して落とす。
        # fp16 デコードで NaN を出した実績のあるモデルだけ fp32 に昇格させる。
        # (デコードは _decode_latents が自前で制御する。このフラグは i2i の encode 側に効く)
        pipe.vae.config.force_upcast = ckpt_name in self._vae_fp32
        if not offload and ckpt_name in self._vae_fp32:
            # 既知の fp16 非対応モデルはロード時点で bf16 に昇格させ、
            # 初回生成の fp16 探り直し (NaN → 撮り直し) を省く
            pipe.vae.to(safe_vae_dtype())
        self._offload = offload
        self._vae_force_tiling = False  # 前のモデルで得た実測の記憶は持ち越さない
        self.current = ckpt_name
        return (
            f"loaded: {ckpt_name}  ({mode})  "
            f"embeddings: {', '.join(self.loaded_embeddings) or 'none'}"
        )

    # ---------- LoRA ----------
    @property
    def loras(self) -> list[tuple[str, float]]:
        """いま載っている LoRA の (名前, 強度)。呼び出し側が書き換えられないよう複製を返す。"""
        return list(self._loras)

    def _load_lora_adapters(self, items, on_phase=None):
        """LoRA をアダプタとして載せる。**オフロードを有効にする前に**呼ぶこと。

        順序が逆だと、オフロードのフックが付いたあとのモジュールに層を足すことに
        なり、デバイス配置が壊れる（重みが CPU と GPU に散らばる）。
        fuse_lora は使わない。融合すると強度を変えるたびに読み直しになるので、
        アダプタのまま持っておき、強度は set_adapters で差し替える。
        """
        self._loras = []
        items = [(n, float(w)) for n, w in items]
        if not items:
            return
        for i, (name, weight) in enumerate(items, 1):
            path = lora.path_of(name)
            if path is None:
                continue  # 手元に無いものは呼び出し側が画面に出す
            _say(on_phase, f"LoRA を読み込んでいます... ({i}/{len(items)} {name})")
            if self._try_load_one(path, name, on_phase):
                self._loras.append((name, weight))
        self._push_adapter_weights()

    def _adapter_registered(self, name: str) -> bool:
        """アダプタが実際に登録されたか。問い合わせられないときは判断しない。"""
        try:
            listed = self.pipe.get_list_adapters()
        except Exception:  # noqa: BLE001
            return True
        return any(name in names for names in listed.values())

    def _try_load_one(self, path, name: str, on_phase=None) -> bool:
        """LoRA を 1 つ載せる。載らなければ理由を知らせて False を返す。

        壊れたファイルや、別の構造のモデル用の LoRA を掴んだだけで生成ごと落ちると、
        何が悪かったのか分からないまま止まる。その 1 つを飛ばして続ける。
        """
        usable, why = lora.inspect(name)
        if not usable:
            print(f"[lora] {name}: {why}")
            _say(on_phase, f"LoRA を使えません: {name}（{why}）")
            return False
        adapter = lora.adapter_name(name)
        try:
            self.pipe.load_lora_weights(str(path), adapter_name=adapter)
        except Exception as e:  # noqa: BLE001 — 読めない理由は問わず、続行を優先する
            # diffusers 0.40 は、テキストエンコーダ側の鍵が片方 (te2) だけ無い LoRA を
            # 読むと、空の rank_dict を掴んで IndexError を投げる。te1 だけを持つ
            # SDXL の LoRA はごく普通にあるので、これで全部を諦めると使えるものが
            # ほとんど無くなる。UNet 側は例外より前に載っているので、アダプタが
            # 登録されていれば「テキストエンコーダ分は当たらなかった」として使う。
            if not self._adapter_registered(adapter):
                print(f"[lora] {name} を読み込めません: {e}")
                _say(on_phase, f"LoRA を読み込めませんでした: {name}")
                return False
            print(f"[lora] {name}: UNet 側のみ適用（テキストエンコーダ分は当たらず: {e}）")
            _say(on_phase, f"LoRA を一部だけ適用しました: {name}")
            return True
        print(f"[lora] {name} を読み込みました")
        if not self._adapter_registered(adapter):
            # 例外は出ないが、当てる先が 1 つも無かった場合。diffusers は警告を出すだけで
            # 進むので、ここで弾かないと後の set_adapters が
            # 「そんなアダプタは無い」で落ちる（実際に踏んだ）
            print(f"[lora] {name}: 当てる先が見つかりませんでした")
            _say(on_phase, f"LoRA を当てられませんでした: {name}")
            return False
        return True

    def _push_adapter_weights(self):
        """いまの (名前, 強度) をパイプラインに反映する。

        i2i パイプは from_pipe で同じ UNet / テキストエンコーダを共有しているので、
        こちらに当てれば両方に効く。
        """
        if not self._loras:
            return
        self.pipe.set_adapters(
            [lora.adapter_name(n) for n, _ in self._loras],
            [w for _, w in self._loras],
        )

    def _swap_lora_adapters(self, items, on_phase=None):
        """常駐時に、載せる LoRA の顔ぶれを入れ替える。

        オフロード中は使えない（上の順序の制約があるため、そちらは作り直す）。
        """
        wanted = [(n, float(w)) for n, w in items if lora.path_of(n) is not None]
        keep = {n for n, _ in wanted}
        gone = [n for n, _ in self._loras if n not in keep]
        if gone:
            self.pipe.delete_adapters([lora.adapter_name(n) for n in gone])
        have = {n for n, _ in self._loras}
        for name, _ in wanted:
            if name in have:
                continue
            _say(on_phase, f"LoRA を読み込んでいます... ({name})")
            if not self._try_load_one(lora.path_of(name), name, on_phase):
                wanted = [x for x in wanted if x[0] != name]
        self._loras = wanted
        if self._loras:
            self._push_adapter_weights()
        else:
            self.pipe.unload_lora_weights()

    def sync_loras(self, items, on_phase=None) -> bool:
        """選択中の LoRA に合わせる。作り直しが要ったかどうかを返す。

        強度だけの違いなら set_adapters で重みを差し替えるだけで済ませる。
        ここでファイルを読み直さないのが肝で、スライダーを動かすたびに
        数十秒待たされることがなくなる。

        顔ぶれが変わったときは、常駐なら差分だけ入れ替える。オフロード中は
        「LoRA を載せる → オフロードを有効にする」の順を守る必要があるため、
        パイプラインごと作り直す（時間がかかるので進捗に出す）。
        """
        if self.pipe is None:
            return False
        items = [(n, float(w)) for n, w in items]
        if not self._lock.acquire(blocking=False):
            _say(on_phase, self._wait_note())
            self._lock.acquire()
        self._busy = "LoRA の切り替え"
        try:
            if [n for n, _ in self._loras] == [n for n, _ in items]:
                if self._loras != items:  # 強度だけが違う
                    self._loras = items
                    self._push_adapter_weights()
                return False
            if self._offload:
                _say(on_phase, "LoRA を入れ替えるためモデルを読み込み直しています...")
                self._load_impl(self.current, on_phase, items, force=True)
                return True
            self._swap_lora_adapters(items, on_phase)
            return False
        finally:
            self._busy = None
            self._lock.release()

    @staticmethod
    def _plan(path: Path, lora_bytes: int = 0) -> tuple[bool, str]:
        """ロード前に VRAM 収支を見て配置を決める。(オフロードするか, 表示用の説明)。

        重い from_single_file を呼ぶ前に判定するので、載らないモデルは待たずに弾ける。
        """
        if not torch.cuda.is_available():
            raise InsufficientVram(
                "CUDA が使える GPU が見つかりません。setup.bat を実行し直してください。"
            )
        # 総容量ではなく「今の空き」で判定する。他のアプリ (ゲーム・録画・dwm) が
        # VRAM を掴んでいると、総容量では載る計算でも実際には共有メモリに溢れて
        # 生成が数十倍遅くなる（進捗バーが 0 のまま動かないように見える）。
        # 空きの取り方は free_vram_bytes を参照。torch の値だけでは他プロセスの
        # 確保が見えず、この判定が素通りしてしまう
        vram = free_vram_bytes() / 1024**3
        # LoRA はチェックポイントとは別に常駐する。判定の式は変えず、
        # 重みの見積りに実ファイルのサイズを足すだけにしてある
        lora_gb = lora_bytes / 1024**3
        size = fp16_footprint(path)
        if size is None:
            # 見積れない形式。弾かずに載せてみるが、VRAM が小さいなら安全側 (オフロード) に倒す
            offload = vram < LOW_VRAM_GB
            return offload, f"空き VRAM {vram:.1f}GB: {'cpu offload (低速)' if offload else 'gpu'}"
        total, unet = size
        total += lora_gb
        unet += lora_gb
        if vram >= total + ACTIVATION_HEADROOM_GB:
            return False, f"空き VRAM {vram:.1f}GB / 重み {total:.1f}GB: gpu"
        if vram >= unet + OFFLOAD_HEADROOM_GB:
            return True, f"空き VRAM {vram:.1f}GB / 重み {total:.1f}GB: cpu offload (低速)"
        raise InsufficientVram(
            f"空き VRAM が足りないため {path.name} は読み込めません。\n"
            f"このモデルは fp16 でも重みが約 {total:.1f}GB、"
            f"オフロードしても最大 {unet:.1f}GB + 作業領域 {OFFLOAD_HEADROOM_GB}GB を"
            f"同時に必要としますが、現在の空き VRAM は {vram:.1f}GB です。\n"
            f"VRAM を使う他のアプリ (ゲーム・配信/録画ソフト・ブラウザ) を閉じてから"
            f"やり直してください。それでも足りなければ、より小さいモデルを使うか、"
            f"engine.py の OFFLOAD_HEADROOM_GB を下げてください。"
        )

    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.pipe_i2i = None  # 共有コンポーネントなので参照を消すだけでよい
            self.current = None
            self.loaded_embeddings = []
            self._loras = []
            self._embed_cache.clear()  # GPU 上のテンソルを持っているので必ず捨てる
            self._compel = None  # 前のモデルのテキストエンコーダを掴んだままにしない
            gc.collect()  # パイプラインは循環参照を持つので、明示的に回収してから
            torch.cuda.empty_cache()

    def warmup(self, width: int, height: int):
        """ロード直後、ユーザーがプロンプトを打っている間に 1 step + デコードを済ませる。

        cuDNN のカーネル探索 (USE_CUDNN_BENCHMARK) と CUDA 初期化を初回生成から追い出す。
        探索結果はテンソル形状ごとにキャッシュされるので、実際に使うプリセット解像度で回す。
        """
        if self.pipe is None or self._offload:
            return  # オフロード時は毎回の転送が支配的で、ウォームアップの意味が薄い
        if not self._lock.acquire(blocking=False):
            return  # 生成が始まっていたら邪魔をしない
        # 画面に出ない裏の処理なので、待たされた側にはこの名前で見える
        self._busy = "次の生成を速くする準備"
        try:
            with torch.inference_mode():
                pe, ne, pp, pn = self._get_embeds("1girl", "", 2)
                lat = self.pipe(
                    prompt_embeds=pe, negative_prompt_embeds=ne,
                    pooled_prompt_embeds=pp, negative_pooled_prompt_embeds=pn,
                    width=width, height=height, num_inference_steps=1,
                    guidance_scale=5.0, output_type="latent",
                    generator=torch.Generator("cuda").manual_seed(0),
                ).images
                self._decode_latents(lat)  # デコード側のカーネルも温める
            print(f"[warmup] {width}x{height} 完了（初回生成のもたつきが消えます）")
        except Exception as e:  # noqa: BLE001 — ウォームアップは失敗しても実害なし
            print(f"[warmup] skip: {e}")
        finally:
            self._busy = None
            self._lock.release()

    # ---------- Embedding (SDXL 形式: clip_l / clip_g) ----------
    def _load_embeddings(self):
        assert self.pipe is not None
        self.loaded_embeddings = []
        for f in sorted(EMB_DIR.glob("*.safetensors")):
            token = f.stem
            try:
                sd = load_file(str(f))
                if "clip_l" in sd and "clip_g" in sd:
                    self.pipe.load_textual_inversion(
                        sd["clip_l"], token=token,
                        text_encoder=self.pipe.text_encoder, tokenizer=self.pipe.tokenizer,
                    )
                    self.pipe.load_textual_inversion(
                        sd["clip_g"], token=token,
                        text_encoder=self.pipe.text_encoder_2, tokenizer=self.pipe.tokenizer_2,
                    )
                    self.loaded_embeddings.append(token)
                else:
                    print(f"[skip] {f.name}: SDXL 形式 (clip_l/clip_g) ではありません")
            except Exception as e:  # noqa: BLE001
                print(f"[warn] embedding {f.name} の読み込みに失敗: {e}")

    # ---------- プロンプト埋め込み ----------
    # UI の Clip Skip (A1111/Civitai と同じ意味) → diffusers の clip_skip 引数。
    #
    # diffusers の SDXL は hidden_states[-(clip_skip + 2)] を取る。つまり clip_skip=None/0 で
    # 既に penultimate = A1111 でいう Clip Skip 2 の層になっている ("SDXL always indexes
    # from the penultimate layer" とソースにある)。したがって差は 2。
    # 旧実装は「diffusers は飛ばす層数を受ける」と誤解して -1 していたため、
    # UI で 2 を選んでも実際には A1111 の Clip Skip 3 相当の層が使われ、
    # Civitai のサンプルと絵柄が一致しなかった (2026-08-24 に実測で確認)。
    #
    #   UI 1 → -1 → hidden_states[-1]   UI 2 → 0 → hidden_states[-2]   UI 3 → 1 → [-3]

    def _get_embeds(self, prompt: str, negative: str, clip_skip: int):
        """encode_prompt の結果を LRU キャッシュ。seed 連打時はテキストエンコードが 0 秒になる。

        オフロード時はさらに効く: キャッシュに当たるとテキストエンコーダ 2 基 (約1.4GB) を
        GPU へ転送する必要そのものが消える。
        """
        key = (self.current, prompt, negative, clip_skip)
        hit = self._embed_cache.get(key)
        if hit is not None:
            self._embed_cache.move_to_end(key)
            return hit
        out = self._encode_compel(prompt, negative, clip_skip)
        if out is None:  # compel が使えない構成 → 素の encode_prompt に落とす
            out = self._encode_plain(prompt, negative, clip_skip)
        self._embed_cache[key] = out
        while len(self._embed_cache) > 8:  # 1 件 ~1.3MB (VRAM)。8 件で十分
            self._embed_cache.popitem(last=False)
        return out

    def _encode_compel(self, prompt: str, negative: str, clip_skip: int):
        """A1111 互換の重み付き構文 `(tag:1.2)` と 77 トークン超のプロンプトを通す。

        CompelForSDXL は penultimate 非正規化 = UI の Clip Skip 2 に固定なので、
        それ以外が選ばれたときは素の encode_prompt に任せる（None を返す）。
        失敗時も None を返し、生成そのものは必ず続行させる。
        """
        if clip_skip != 2:
            return None
        try:
            c = self._compel
            if c is None:
                from compel import CompelForSDXL

                c = CompelForSDXL(self.pipe, device=self.pipe._execution_device)
                self._compel = c
            r = c(prompt, negative_prompt=negative or "")
            return r.embeds, r.negative_embeds, r.pooled_embeds, r.negative_pooled_embeds
        except Exception as e:  # noqa: BLE001 — 構文エラー等で生成を殺さない
            print(f"[prompt] compel が使えないため通常解釈にフォールバック: {e}")
            return None

    def _encode_plain(self, prompt: str, negative: str, clip_skip: int):
        return self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.pipe._execution_device,
            num_images_per_prompt=1,  # 枚数方向の複製はパイプライン側に任せる
            do_classifier_free_guidance=True,
            negative_prompt=negative,
            clip_skip=clip_skip - 2,  # 上の注記を参照
        )

    # ---------- VAE デコード ----------
    @staticmethod
    @contextlib.contextmanager
    def _no_cudnn_benchmark():
        """VAE デコードの間だけ cuDNN のカーネル探索を止める。

        benchmark は「同じ形状を何度も回す」UNet では元が取れるが、1 生成 1 回の
        VAE デコードでは、832x1216 級の巨大 conv の総当たり探索に数十秒かかる
        (73 秒を実測) 割に縮むのは 0.1 秒級で大損。ヒューリスティック選択で即決させる。
        """
        prev = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        try:
            yield
        finally:
            torch.backends.cudnn.benchmark = prev

    def _plan_tiling(self, lat: torch.Tensor) -> None:
        """このデコードでタイル分割を使うかを毎回決め、明示的に切り替える。

        一括デコードは、VRAM の空きが足りないと OOM 例外を出さずに共有メモリへ
        溢れて数十秒〜数分級になる (Windows の仕様)。例外で捕まえられないので、
        必要量を見積もって足りなければタイル分割へ落とす。
        必要量はデコード活性のピーク実測から: おおよそ 5500 bytes/出力ピクセル
        (fp16/bf16。4400 では判定を通ったのに peak 13.5GB まで溢れたため保守的に)

        毎回決め直すのが要点。以前はここで enable_tiling() を呼ぶだけで戻す側が
        無かったため、他アプリが一時的に VRAM を掴んでいた等で一度でも分割に入ると、
        モデルを読み直すまでずっと分割のままだった。常駐時は一括の方が速く、
        タイル境界の継ぎ目も出ないので、その状態に自力で戻れるようにする。
        """
        vae = self.pipe.vae
        pixels = lat.shape[-2] * lat.shape[-1] * 64  # latent 1 画素 = 出力 8x8 画素
        need = pixels * (11000 if vae.dtype == torch.float32 else 5500)
        free = torch.cuda.mem_get_info()[0]
        avail = free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()

        # オフロード時は見積りに関係なく常に分割する。ロード判定の OFFLOAD_HEADROOM_GB
        # (1.5GB) が「タイル分割デコード前提」で常駐時より小さく取ってあるため、
        # ここで一括に戻すとロードを通した前提そのものが崩れる。
        note = ""
        if self._offload:
            tile, note = True, " / オフロード中は常に分割"
        elif self._vae_force_tiling:
            tile, note = True, " / 実測で不足と判明済み"
        else:
            tile = avail < need
        if tile:
            vae.enable_tiling()
        else:
            vae.disable_tiling()
        print(f"[vae] {'タイル分割' if tile else '一括'}デコード "
              f"(必要 ~{need / 1024**3:.1f}GB / 利用可 {avail / 1024**3:.1f}GB{note})")

    def _decode_latents(self, latents: torch.Tensor, on_phase=None) -> list[Image.Image]:
        """latent を 1 枚ずつ画像へ。fp16 一括デコードを基本に、二段の安全網を持つ。

        - fp16 の結果に NaN/Inf → そのモデルを bf16 デコードに切り替えて撮り直し。
          NaN の原因は fp16 の値域あふれで、bf16 は fp32 と同じ値域を持つため出ない。
          速度は fp16 並み (fp32 の約 4 倍速)、fp32 との画素差は平均 0.3/255 で不可視 (実測)
        - VRAM に載らない (OOM) → タイル分割を有効にして撮り直し
        1 枚ずつ回すのはバッチ n 枚のピークを 1 枚分に抑えるため (enable_slicing 相当)。
        """
        pipe = self.pipe
        vae = pipe.vae
        cfg = vae.config
        if getattr(cfg, "latents_mean", None) and getattr(cfg, "latents_std", None):
            mean = torch.tensor(cfg.latents_mean).view(1, 4, 1, 1).to(latents)
            std = torch.tensor(cfg.latents_std).view(1, 4, 1, 1).to(latents)
            lat = latents * std / cfg.scaling_factor + mean
        else:
            lat = latents / cfg.scaling_factor
        if self.current in self._vae_fp32 and vae.dtype == torch.float16:
            # 既知の fp16 非対応モデル。最初から安全な dtype (bf16、無ければ fp32) で回す
            vae.to(safe_vae_dtype())
        self._plan_tiling(lat)
        pils: list[Image.Image] = []
        with self._no_cudnn_benchmark():
            pils = self._decode_loop(lat, pils, on_phase)
        # デコードで一時的に膨らんだ予約が物理 VRAM を超えたままだと、次の生成が
        # 共有メモリ退避の影響を引きずって数割遅くなる。超過時のみ返却する
        props = torch.cuda.get_device_properties(0)
        if torch.cuda.memory_reserved() > props.total_memory * 0.9:
            torch.cuda.empty_cache()
        return pils

    def _decode_loop(self, lat, pils, on_phase=None) -> list[Image.Image]:
        pipe = self.pipe
        vae = pipe.vae
        i = 0
        while i < lat.shape[0]:
            one = lat[i:i + 1]
            try:
                img = vae.decode(one.to(vae.dtype), return_dict=False)[0]
                if vae.dtype != torch.float32 and bool(
                    torch.isnan(img).any() | torch.isinf(img).any()
                ):
                    if vae.dtype == torch.float16:
                        # fp16 で壊れる VAE。このモデルは以後 bf16 でデコードする
                        self._vae_fp32.add(self.current)
                        try:  # 次回起動から探り直さないよう記録（失敗しても動作に影響なし）
                            VAE_FP32_FILE.write_text(
                                json.dumps(sorted(self._vae_fp32), ensure_ascii=False),
                                encoding="utf-8",
                            )
                        except OSError:
                            pass
                        vae.config.force_upcast = True
                        alt = safe_vae_dtype()
                        vae.to(alt)
                        name = "bf16" if alt == torch.bfloat16 else "fp32"
                        print(f"[vae] {self.current}: fp16 デコードが NaN/Inf を出したため {name} に切替")
                        _say(on_phase, f"{name} で変換をやり直しています...")
                    else:
                        # bf16 でも NaN (理論上ほぼ起きない)。最後の砦の fp32 + タイル分割へ
                        vae.to(torch.float32)
                        vae.enable_tiling()
                        self._vae_force_tiling = True
                        print(f"[vae] {self.current}: bf16 でも NaN/Inf のため fp32 (タイル分割) に切替")
                        _say(on_phase, "変換をやり直しています...")

                    continue
            except torch.cuda.OutOfMemoryError:
                if getattr(vae, "use_tiling", False):
                    raise  # タイル分割でも溢れた。これ以上は退避先がない
                print("[vae] 一括デコードが VRAM に載らないためタイル分割に切替")
                _say(on_phase, "VRAM が足りないため分割して変換しています...")
                torch.cuda.empty_cache()
                vae.enable_tiling()
                self._vae_force_tiling = True  # 見積りが外れた。以後は毎回の判定より優先
                continue
            pils += pipe.image_processor.postprocess(img.float(), output_type="pil")
            i += 1
        # 注: bf16 に昇格した VAE は fp16 へ戻さない。戻すと毎生成で往復キャストを払う上、
        # どうせ次の生成でまた bf16 が必要になる。サイズは fp16 と同じで常駐コストもない
        return pils

    # ---------- 計測 ----------
    @staticmethod
    @contextlib.contextmanager
    def _phase(sink: dict, name: str):
        """工程別タイマー。CUDA は非同期なので synchronize してから測る。"""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize()
        sink[name] = time.perf_counter() - t0

    @staticmethod
    def _perf_line(phases: dict, steps: int) -> str:
        peak = torch.cuda.max_memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        parts = [f"{k} {v:.2f}s" for k, v in phases.items()]
        if steps and "unet" in phases:
            parts.append(f"{phases['unet'] / steps:.2f}s/step")
        parts.append(f"peak {peak:.1f}GB")
        line = " | ".join(parts)
        if peak > total * 0.92:
            line += "  ⚠ VRAM 上限付近 — 共有メモリ退避で激遅になっている可能性"
        return line

    @contextlib.contextmanager
    def _attention_slicing_for(self, width: int, height: int):
        """高解像度のときだけアテンションを分割する。

        SDXL の活性値は画素数にほぼ比例するので、2 倍 (画素数 4 倍) では 12GB に載らない。
        分割すると速度は落ちるが、溢れて共有メモリに退避するよりは桁違いに速い。
        通常解像度では何もしない (分割は純粋なオーバーヘッドなので)。
        """
        need = width * height > ATTENTION_SLICING_PIXELS
        if not need:
            yield
            return
        self.pipe.enable_attention_slicing()
        try:
            yield
        finally:
            self.pipe.disable_attention_slicing()

    # ---------- 高解像度化 (Hires.fix) ----------
    def upscale(self, image, prompt: str, negative: str, cfg: float, sampler: str,
                clip_skip: int, seed: int, scale: float = HIRES_SCALE,
                strength: float = HIRES_STRENGTH, steps: int = HIRES_STEPS,
                step_cb=None):
        """生成済み画像を拡大し、img2img で細部を描き足す。

        返り値は generate と同じ (images | None, seed, 計測行)。
        拡大自体は img2img 経路が入力画像をリサイズして行う。
        """
        assert self.pipe is not None, "model not loaded"
        w = max(8, int(image.width * scale) // 8 * 8)
        h = max(8, int(image.height * scale) // 8 * 8)
        return self.generate(
            prompt, negative, w, h, steps, cfg, sampler, clip_skip, seed, 1,
            image=image, strength=strength, step_cb=step_cb,
        )

    # ---------- 変分 (この絵を少しだけ変える) ----------
    def variation_latents(self, seed: int, width: int, height: int, count: int,
                          strength: float = VARIATION_STRENGTH):
        """元の seed のノイズを保ちつつ、少しだけずらしたノイズを count 個作る。

        A1111 の variation seed と同じ考え方。元ノイズと別 seed のノイズを
        球面補間で混ぜるので、構図や色を残したまま細部だけが変わる。
        """
        dev = self.pipe._execution_device
        ch = self.pipe.unet.config.in_channels
        shape = (1, ch, height // 8, width // 8)
        dtype = self.pipe.unet.dtype
        g = torch.Generator(device=dev).manual_seed(int(seed))
        base = torch.randn(shape, generator=g, device=dev, dtype=dtype)
        mixed = []
        for i in range(count):
            # 変分側の seed は元 seed から決定的に導く（同じ絵からは毎回同じ枚数分が出る）
            gv = torch.Generator(device=dev).manual_seed(int(seed) + 100003 * (i + 1))
            v = torch.randn(shape, generator=gv, device=dev, dtype=dtype)
            mixed.append(slerp(strength, base, v))
        # init_noise_sigma は掛けない。diffusers の prepare_latents が、渡した latents にも
        # 必ず掛けるため（pipeline_stable_diffusion_xl.py の "scale the initial noise" の行）。
        # ここで掛けると二乗になり、Euler 系では約 214 倍のノイズになって絵が壊れる。
        return torch.cat(mixed, dim=0)

    def _variation_generators(self, seed: int, width: int, height: int, n: int):
        """変分用の generator を、通常経路と同じ乱数位置に揃えて作る。

        euler_a などの ancestral 系サンプラは毎ステップ generator からノイズを引く。
        通常経路では prepare_latents が初期ノイズ生成で generator を 1 回消費するが、
        変分では latents を自前で渡すため消費されない。そのままだと以降のノイズ列が
        丸ごとずれ、元の絵とまったく似ない絵が出る（実測で相関 +0.15 まで落ちた）。
        ここで同じ形の randn を 1 回空引きして位置を合わせる。

        さらに、全変分で同じ seed を使う。こうすると各ステップのノイズが全変分で共通になり、
        違いは初期 latents だけになる = 元の絵を保ったまま細部だけ変わる。
        """
        dev = self.pipe._execution_device
        ch = self.pipe.unet.config.in_channels
        shape = (1, ch, height // 8, width // 8)
        dtype = self.pipe.unet.dtype
        gens = []
        for _ in range(n):
            g = torch.Generator(device=dev).manual_seed(int(seed))
            torch.randn(shape, generator=g, device=dev, dtype=dtype)  # 位置合わせの空引き
            gens.append(g)
        return gens

    def variations(self, prompt: str, negative: str, width: int, height: int, steps: int,
                   cfg: float, sampler: str, clip_skip: int, seed: int,
                   count: int = VARIATION_COUNT, strength: float = VARIATION_STRENGTH,
                   step_cb=None, on_phase=None):
        """気に入った絵の seed を固定したまま、少しだけ違う絵を count 枚出す。"""
        assert self.pipe is not None, "model not loaded"
        lat = self.variation_latents(seed, width, height, count, strength)
        return self.generate(
            prompt, negative, width, height, steps, cfg, sampler, clip_skip, seed, count,
            step_cb=step_cb, init_latents=lat, on_phase=on_phase,
        )

    # ---------- 生成 ----------
    def set_sampler(self, name: str):
        assert self.pipe is not None
        sched = SAMPLERS[name](self.pipe.scheduler.config)
        self.pipe.scheduler = sched
        if self.pipe_i2i is not None:  # scheduler の差し替えは両パイプに反映する
            self.pipe_i2i.scheduler = sched

    def request_cancel(self):
        """UI の中止ボタン。次の step で生成を打ち切る（途中の絵は保存されない）。"""
        self.cancel.set()

    def generate(self, *args, on_phase=None, **kwargs):
        # 生成も直列化する。二重クリック等で 2 つ同時に走ると作業メモリが倍増して溢れる
        if not self._lock.acquire(blocking=False):
            _say(on_phase, self._wait_note())
            self._lock.acquire()
        self._busy = "画像の生成"
        try:
            return self._generate_impl(*args, on_phase=on_phase, **kwargs)
        finally:
            self._busy = None
            self._lock.release()

    @torch.inference_mode()
    def _generate_impl(
        self,
        prompt: str,
        negative: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        sampler: str,
        clip_skip: int,
        seed: int,
        n: int = 1,
        image=None,
        strength: float = 0.6,
        step_cb=None,
        init_latents=None,
        on_phase=None,
    ):
        """返り値: (images | None, seed, 計測行)。中止されたときは images が None。"""
        assert self.pipe is not None, "model not loaded"
        self.cancel.clear()
        torch.cuda.reset_peak_memory_stats()
        phases: dict[str, float] = {}
        self.set_sampler(sampler)
        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        if init_latents is None:
            gens = [torch.Generator("cuda").manual_seed(seed + i) for i in range(n)]
        else:
            gens = self._variation_generators(seed, width, height, n)

        with self._phase(phases, "embed"):
            pe, ne, pp, pn = self._get_embeds(prompt, negative, clip_skip)

        # i2i の実効 step 数は steps × strength (diffusers の仕様)
        total = steps if image is None else max(1, int(steps * strength))

        def cb(pipe, i, t, kw):
            # 毎 step 呼ばれる。中止の判定と、進捗バーへの step 数の通知だけを行う。
            # ここは denoise ループの中なので、重い処理を足すとそのまま生成時間に乗る
            if self.cancel.is_set():
                pipe._interrupt = True  # 残りのループを空回りさせて即座に抜ける
            elif step_cb is not None:
                step_cb(i + 1, total)
            return {}

        common = dict(
            prompt_embeds=pe, negative_prompt_embeds=ne,
            pooled_prompt_embeds=pp, negative_pooled_prompt_embeds=pn,
            num_images_per_prompt=n,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=gens,
            output_type="latent",  # デコードは _decode_latents で自前制御する
            callback_on_step_end=cb,
        )
        with self._phase(phases, "unet"), self._attention_slicing_for(width, height):
            if image is not None:  # 入力画像があれば img2img
                # img2img パイプは width/height を受け取らず (**kwargs に吸われて無視され)、
                # 出力サイズは入力画像のサイズそのものになる。狙った解像度で出すには
                # 渡す前に自前でリサイズする必要がある。Hires.fix もこの経路で拡大する
                src = image.convert("RGB")
                if src.size != (width, height):
                    src = src.resize((width, height), Image.LANCZOS)
                latents = self.pipe_i2i(image=src, strength=strength, **common).images
            else:
                if init_latents is not None:  # 変分: 混ぜた初期ノイズを指定する
                    common["latents"] = init_latents
                latents = self.pipe(width=width, height=height, **common).images

        if self.cancel.is_set():
            return None, seed, self._perf_line(phases, 0)  # 中断: デコード代も払わない

        _say(on_phase, "画像に変換しています...")
        with self._phase(phases, "vae"):
            images = self._decode_latents(latents, on_phase)
        return images, seed, self._perf_line(phases, total)
