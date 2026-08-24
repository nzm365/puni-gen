"""diffusers ベースの SDXL/Illustrious 推論エンジン。ComfyUI 非依存。"""
from __future__ import annotations

import contextlib
import gc
import json
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
CKPT_DIR = ROOT / "models" / "checkpoints"
EMB_DIR = ROOT / "models" / "embeddings"
PRESET_DIR = ROOT / "presets"

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

# 何ステップごとに途中プレビューを出すか
PREVIEW_EVERY = 3

# latent → RGB の線形近似係数 (ComfyUI が SDXL プレビューに使っているもの)。
# TAESDXL が使えない環境向けの、依存ゼロ・ほぼゼロコストのフォールバック
_LATENT_RGB = torch.tensor([
    [ 0.3920,  0.4054,  0.4549],
    [-0.2634, -0.0196,  0.0653],
    [ 0.0568,  0.1687, -0.0755],
    [-0.3112, -0.2359, -0.2076],
])

# 単一ファイル形式のチェックポイントで UNet が持つキーの接頭辞。
# オフロード時は「一番大きい単体モジュール = UNet」が VRAM ピークになる。
UNET_PREFIX = "model.diffusion_model."


class InsufficientVram(RuntimeError):
    """VRAM が足りないと分かったのでロードを中止した。"""


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
        self._offload = False
        # (モデル, プロンプト, ネガティブ, clip_skip) → encode_prompt の結果。
        # seed だけ変えて連打する使い方ではテキストエンコーダ 2 基を毎回回す必要がない
        self._embed_cache: OrderedDict[tuple, tuple] = OrderedDict()
        # fp16 デコードで NaN/Inf を出した実績のあるモデル名 (以後そのモデルは fp32 でデコード)
        self._vae_fp32: set[str] = set()
        self._taesd = None  # AutoencoderTiny | "unavailable" | None (未初期化)
        self._taesd_started = False
        if torch.cuda.is_available():
            if USE_CUDNN_BENCHMARK:
                torch.backends.cudnn.benchmark = True
            if USE_TF32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    # ---------- モデル ----------
    def load(self, ckpt_name: str) -> str:
        with self._lock:
            return self._load_impl(ckpt_name)

    def _load_impl(self, ckpt_name: str) -> str:
        if self.current == ckpt_name and self.pipe is not None:
            return f"already loaded: {ckpt_name}"
        self.unload()
        path = CKPT_DIR / ckpt_name
        offload, mode = self._plan(path)  # 載らないと分かればここで例外
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(path), torch_dtype=torch.float16, use_safetensors=True
        )
        self.pipe = pipe
        self._load_embeddings()  # オフロードのフックが付く前に済ませる
        if USE_FUSE_QKV:
            pipe.fuse_qkv_projections()  # attention の Q/K/V を 1 つの行列積に融合 (+1.2GB)
        # i2i パイプは t2i と全コンポーネントを共有する（重みのコピーは発生しない）。
        # オフロードのフックが付く前に作っておく。
        # torch_dtype は必ず明示する。diffusers 0.40 の from_pipe は dtype 未指定だと
        # 共有コンポーネントごと fp32 へキャストするため、重みが倍の 13GB になり
        # VRAM から溢れて生成が数十倍遅くなる（実測済み）。
        self.pipe_i2i = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe, torch_dtype=torch.float16)
        if offload:
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()  # 8GB 級では VAE デコードのピークも惜しい
        else:
            pipe.to("cuda")
            if USE_CHANNELS_LAST:
                # Tensor Core が効きやすいメモリ配置（オフロード時は転送が支配的なので省略）
                pipe.unet.to(memory_format=torch.channels_last)
                pipe.vae.to(memory_format=torch.channels_last)
            # 常駐時はタイル分割しない。プリセット解像度 (832x1216 / 1024x1536) は
            # 旧実装ではどちらも常にタイル分割を踏んでいた（しきい値が 1024px のため）。
            # 一括デコードの方が速く、タイル境界の継ぎ目も原理的に消える。
            # 万一 VRAM に載らなければ _decode_latents が OOM を検出して自動でタイル分割に落とす。
        # fp16 デコードで NaN を出した実績のあるモデルだけ fp32 に昇格させる。
        # (デコードは _decode_latents が自前で制御する。このフラグは i2i の encode 側に効く)
        pipe.vae.config.force_upcast = ckpt_name in self._vae_fp32
        self._offload = offload
        self.current = ckpt_name
        if not self._taesd_started:  # プレビュー用の軽量 VAE を裏で用意しておく
            self._taesd_started = True
            threading.Thread(target=self._init_taesd, daemon=True).start()
        return (
            f"loaded: {ckpt_name}  ({mode})  "
            f"embeddings: {', '.join(self.loaded_embeddings) or 'none'}"
        )

    @staticmethod
    def _plan(path: Path) -> tuple[bool, str]:
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
        vram = torch.cuda.mem_get_info()[0] / 1024**3
        size = fp16_footprint(path)
        if size is None:
            # 見積れない形式。弾かずに載せてみるが、VRAM が小さいなら安全側 (オフロード) に倒す
            offload = vram < LOW_VRAM_GB
            return offload, f"空き VRAM {vram:.1f}GB: {'cpu offload (低速)' if offload else 'gpu'}"
        total, unet = size
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
            self._embed_cache.clear()  # GPU 上のテンソルを持っているので必ず捨てる
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
        out = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.pipe._execution_device,
            num_images_per_prompt=1,  # 枚数方向の複製はパイプライン側に任せる
            do_classifier_free_guidance=True,
            negative_prompt=negative,
            clip_skip=clip_skip - 1 if clip_skip > 1 else None,  # diffusers は "飛ばす層数" を受ける
        )
        self._embed_cache[key] = out
        while len(self._embed_cache) > 8:  # 1 件 ~1.3MB (VRAM)。8 件で十分
            self._embed_cache.popitem(last=False)
        return out

    # ---------- VAE デコード ----------
    def _decode_latents(self, latents: torch.Tensor) -> list[Image.Image]:
        """latent を 1 枚ずつ画像へ。fp16 一括デコードを基本に、二段の安全網を持つ。

        - fp16 の結果に NaN/Inf → そのモデルを fp32 デコードに切り替えて撮り直し
          (2023 年型の fp16 非対応 VAE への対処。Illustrious 系マージはほぼ fp16 安全版)
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
        use_fp32 = self.current in self._vae_fp32
        if use_fp32:
            pipe.upcast_vae()
        pils: list[Image.Image] = []
        i = 0
        try:
            while i < lat.shape[0]:
                one = lat[i:i + 1]
                try:
                    if use_fp32:
                        img = vae.decode(one.to(torch.float32), return_dict=False)[0]
                    else:
                        img = vae.decode(one.to(vae.dtype), return_dict=False)[0]
                        if bool(torch.isnan(img).any() | torch.isinf(img).any()):
                            # fp16 で壊れる VAE。このモデルは以後 fp32 でデコードする
                            self._vae_fp32.add(self.current)
                            vae.config.force_upcast = True
                            use_fp32 = True
                            pipe.upcast_vae()
                            print(f"[vae] {self.current}: fp16 デコードが NaN/Inf を出したため fp32 に切替")
                            continue
                except torch.cuda.OutOfMemoryError:
                    if getattr(vae, "use_tiling", False):
                        raise  # タイル分割でも溢れた。これ以上は退避先がない
                    print("[vae] 一括デコードが VRAM に載らないためタイル分割に切替")
                    torch.cuda.empty_cache()
                    vae.enable_tiling()
                    continue
                pils += pipe.image_processor.postprocess(img.float(), output_type="pil")
                i += 1
        finally:
            if use_fp32:
                vae.to(torch.float16)  # 次の生成に fp32 の VAE を残さない
        return pils

    # ---------- 途中プレビュー ----------
    def _init_taesd(self):
        """プレビュー用の軽量 VAE (TAESDXL, 約10MB) を裏で取得する。

        初回のみ Hugging Face から落とす。オフライン等で取得できなければ
        線形近似プレビュー (_LATENT_RGB) に自動で落ちる。生成には影響しない。
        """
        try:
            from diffusers import AutoencoderTiny
            dec = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
            self._taesd = dec.to("cuda")
            print("[preview] TAESDXL を読み込みました（高精細プレビュー）")
        except Exception as e:  # noqa: BLE001
            self._taesd = "unavailable"
            print(f"[preview] TAESDXL が使えないため簡易プレビューで動きます: {e}")

    def _preview_images(self, latents: torch.Tensor, size: tuple[int, int]) -> list[Image.Image]:
        """現在の latent から途中経過の画像を作る。数 ms 級で、本生成の品質には無関係。"""
        w, h = max(size[0] // 2, 8), max(size[1] // 2, 8)  # プレビューは半分の解像度で十分
        dec = self._taesd
        if dec is not None and not isinstance(dec, str):
            img = dec.decode(latents.to(dec.dtype)).sample
            pils = self.pipe.image_processor.postprocess(img.float(), output_type="pil")
            return [im.resize((w, h), Image.BILINEAR) for im in pils]
        m = _LATENT_RGB.to(latents.device, latents.dtype)
        rgb = torch.einsum("nchw,cr->nrhw", latents, m)
        rgb = ((rgb + 1) / 2).clamp(0, 1).mul(255).round().byte().cpu()
        return [
            Image.fromarray(t.permute(1, 2, 0).numpy()).resize((w, h), Image.BILINEAR)
            for t in rgb
        ]

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

    def generate(self, *args, **kwargs):
        # 生成も直列化する。二重クリック等で 2 つ同時に走ると作業メモリが倍増して溢れる
        with self._lock:
            return self._generate_impl(*args, **kwargs)

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
        preview_cb=None,
    ):
        """返り値: (images | None, seed, 計測行)。中止されたときは images が None。"""
        assert self.pipe is not None, "model not loaded"
        self.cancel.clear()
        torch.cuda.reset_peak_memory_stats()
        phases: dict[str, float] = {}
        self.set_sampler(sampler)
        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        gens = [torch.Generator("cuda").manual_seed(seed + i) for i in range(n)]

        with self._phase(phases, "embed"):
            pe, ne, pp, pn = self._get_embeds(prompt, negative, clip_skip)

        # i2i の実効 step 数は steps × strength (diffusers の仕様)
        total = steps if image is None else max(1, int(steps * strength))

        def cb(pipe, i, t, kw):
            if self.cancel.is_set():
                pipe._interrupt = True  # 残りのループを空回りさせて即座に抜ける
            elif preview_cb is not None:
                done = i + 1
                pv = None
                if done % PREVIEW_EVERY == 0 and done < total:
                    try:
                        pv = self._preview_images(kw["latents"], (width, height))
                    except Exception:  # noqa: BLE001 — プレビュー失敗で生成を殺さない
                        pv = None
                preview_cb(done, total, pv)
            return {}

        common = dict(
            prompt_embeds=pe, negative_prompt_embeds=ne,
            pooled_prompt_embeds=pp, negative_pooled_prompt_embeds=pn,
            num_images_per_prompt=n,
            width=width, height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=gens,
            output_type="latent",  # デコードは _decode_latents で自前制御する
            callback_on_step_end=cb,
        )
        with self._phase(phases, "unet"):
            if image is not None:  # 入力画像があれば img2img
                latents = self.pipe_i2i(image=image.convert("RGB"), strength=strength, **common).images
            else:
                latents = self.pipe(**common).images

        if self.cancel.is_set():
            return None, seed, self._perf_line(phases, 0)  # 中断: デコード代も払わない

        with self._phase(phases, "vae"):
            images = self._decode_latents(latents)
        return images, seed, self._perf_line(phases, total)
