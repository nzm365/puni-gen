"""diffusers ベースの SDXL/Illustrious 推論エンジン。ComfyUI 非依存。"""
from __future__ import annotations

import json
from pathlib import Path

import torch
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

# 重みサイズを見積れなかったときのフォールバック閾値。
# 一般的な SDXL (fp16 で約 6.5GB) が常駐できるかどうかの目安。
LOW_VRAM_GB = 10

# 速度優先の数値設定（cuDNN autotune / TF32 / channels_last / QKV 融合）。
# 見た目の品質は変わらないが、ビット単位では結果が変わりうる。
# 同じ seed で完全に同じピクセルを再現したいときは False にする。
SPEED_TWEAKS = True

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
        if SPEED_TWEAKS and torch.cuda.is_available():
            # 解像度が数種類に固定される使い方では cuDNN の autotune が効く
            # （新しい解像度の最初の 1 回だけ、カーネル探索の分わずかに遅い）
            torch.backends.cudnn.benchmark = True
            # fp32 行列積を TF32 で。効くのは fp32 に自動昇格される VAE デコードあたり
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ---------- モデル ----------
    def load(self, ckpt_name: str) -> str:
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
        if SPEED_TWEAKS:
            pipe.fuse_qkv_projections()  # attention の Q/K/V を 1 つの行列積に融合
        # i2i パイプは t2i と全コンポーネントを共有する（重みのコピーは発生しない）。
        # オフロードのフックが付く前に作っておく
        self.pipe_i2i = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
        if offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
            if SPEED_TWEAKS:
                # Tensor Core が効きやすいメモリ配置（オフロード時は転送が支配的なので省略）
                pipe.unet.to(memory_format=torch.channels_last)
                pipe.vae.to(memory_format=torch.channels_last)
        # 高解像度でも VAE でメモリを食い潰さない（i2i パイプとは VAE を共有しているので一度でよい）
        pipe.vae.enable_tiling()
        self.current = ckpt_name
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
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        size = fp16_footprint(path)
        if size is None:
            # 見積れない形式。弾かずに載せてみるが、VRAM が小さいなら安全側 (オフロード) に倒す
            offload = vram < LOW_VRAM_GB
            return offload, f"VRAM {vram:.0f}GB: {'cpu offload (低速)' if offload else 'gpu'}"
        total, unet = size
        if vram >= total + ACTIVATION_HEADROOM_GB:
            return False, f"VRAM {vram:.0f}GB / 重み {total:.1f}GB: gpu"
        if vram >= unet + ACTIVATION_HEADROOM_GB:
            return True, f"VRAM {vram:.0f}GB / 重み {total:.1f}GB: cpu offload (低速)"
        raise InsufficientVram(
            f"VRAM が足りないため {path.name} は読み込めません。\n"
            f"このモデルは fp16 でも重みが約 {total:.1f}GB、"
            f"オフロードしても最大 {unet:.1f}GB + 作業領域 {ACTIVATION_HEADROOM_GB}GB を"
            f"同時に必要としますが、この GPU の VRAM は {vram:.1f}GB です。\n"
            f"より小さいモデルを使うか、engine.py の ACTIVATION_HEADROOM_GB を下げてください。"
        )

    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.pipe_i2i = None  # 共有コンポーネントなので参照を消すだけでよい
            self.current = None
            self.loaded_embeddings = []
            torch.cuda.empty_cache()

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

    # ---------- 生成 ----------
    def set_sampler(self, name: str):
        assert self.pipe is not None
        sched = SAMPLERS[name](self.pipe.scheduler.config)
        self.pipe.scheduler = sched
        if self.pipe_i2i is not None:  # scheduler の差し替えは両パイプに反映する
            self.pipe_i2i.scheduler = sched

    @torch.inference_mode()
    def generate(
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
    ):
        assert self.pipe is not None, "model not loaded"
        self.set_sampler(sampler)
        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        gens = [torch.Generator("cuda").manual_seed(seed + i) for i in range(n)]
        kwargs = dict(
            prompt=[prompt] * n,
            negative_prompt=[negative] * n,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            clip_skip=clip_skip - 1 if clip_skip > 1 else None,  # diffusers は "飛ばす層数" を受ける
            generator=gens,
        )
        if image is not None:  # 入力画像があれば img2img
            out = self.pipe_i2i(image=image.convert("RGB"), strength=strength, **kwargs)
        else:
            out = self.pipe(**kwargs)
        return out.images, seed
