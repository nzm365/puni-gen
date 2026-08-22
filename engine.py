"""diffusers ベースの SDXL/Illustrious 推論エンジン。ComfyUI 非依存。"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from diffusers import (
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / "models" / "checkpoints"
EMB_DIR = ROOT / "models" / "embeddings"
PRESET_DIR = ROOT / "presets"

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
        self.current: str | None = None
        self.loaded_embeddings: list[str] = []

    # ---------- モデル ----------
    def load(self, ckpt_name: str) -> str:
        if self.current == ckpt_name and self.pipe is not None:
            return f"already loaded: {ckpt_name}"
        self.unload()
        path = CKPT_DIR / ckpt_name
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(path), torch_dtype=torch.float16, use_safetensors=True
        )
        pipe.to("cuda")
        pipe.enable_vae_tiling()  # 1024x1536 でも VRAM 12GB で安全に
        self.pipe = pipe
        self.current = ckpt_name
        self._load_embeddings()
        return f"loaded: {ckpt_name}  embeddings: {', '.join(self.loaded_embeddings) or 'none'}"

    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
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
        self.pipe.scheduler = SAMPLERS[name](self.pipe.scheduler.config)

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
    ):
        assert self.pipe is not None, "model not loaded"
        self.set_sampler(sampler)
        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        gens = [torch.Generator("cuda").manual_seed(seed + i) for i in range(n)]
        out = self.pipe(
            prompt=[prompt] * n,
            negative_prompt=[negative] * n,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            clip_skip=clip_skip - 1 if clip_skip > 1 else None,  # diffusers は "飛ばす層数" を受ける
            generator=gens,
        )
        return out.images, seed
