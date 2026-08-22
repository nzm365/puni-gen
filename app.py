"""最小構成の Gradio UI。モデルを切り替えるとプリセット（推奨設定）が自動で入る。"""
from datetime import datetime
from pathlib import Path

import gradio as gr

from engine import Engine, InsufficientVram, SAMPLERS, fit_size, list_checkpoints, load_preset

OUT_DIR = Path(__file__).parent / "outputs"
engine = Engine()

ASPECTS = {
    "縦 (プリセット)": None,
    "縦 832x1216": (832, 1216),
    "縦 1024x1536": (1024, 1536),
    "横 1216x832": (1216, 832),
    "横 1536x1024": (1536, 1024),
    "正方 1024x1024": (1024, 1024),
}


def load_or_alert(ckpt) -> str:
    """VRAM 不足はトレースバックではなく画面のエラー表示にする。"""
    try:
        return engine.load(ckpt)
    except InsufficientVram as e:
        raise gr.Error(str(e)) from e


def on_model_change(ckpt):
    """モデル選択 → ロード＋プリセット適用"""
    if not ckpt:
        return [gr.update()] * 7
    msg = load_or_alert(ckpt)
    p = load_preset(ckpt)
    return (
        msg,
        p["prefix_pos"],
        p["default_neg"],
        p["steps"],
        p["cfg"],
        p["sampler"],
        p["clip_skip"],
    )


def on_generate(ckpt, prefix, prompt, negative, aspect, n, steps, cfg, sampler, clip_skip, seed,
                in_image, strength):
    if engine.current != ckpt:
        load_or_alert(ckpt)
    size = ASPECTS[aspect] or tuple(load_preset(ckpt)["size"])
    if in_image is not None:
        # i2i では入力のアスペクト比を保ち、選択サイズは画素数の目安として使う
        size = fit_size(in_image.size, size[0] * size[1])
    full_prompt = (prefix or "") + prompt
    images, used_seed = engine.generate(
        full_prompt, negative, size[0], size[1],
        int(steps), float(cfg), sampler, int(clip_skip), int(seed), int(n),
        image=in_image, strength=float(strength),
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, im in enumerate(images):
        im.save(OUT_DIR / f"{stamp}_{used_seed + i}.png")
    mode = f"i2i (strength {strength})" if in_image is not None else "t2i"
    return images, f"seed: {used_seed}  size: {size[0]}x{size[1]}  mode: {mode}  prompt: {full_prompt}"


with gr.Blocks(title="RTX Easy Image Gen") as demo:
    ckpts = list_checkpoints()
    gr.Markdown("## RTX Easy Image Gen")
    with gr.Row():
        model_dd = gr.Dropdown(ckpts, value=ckpts[0] if ckpts else None, label="モデル", scale=4)
        refresh = gr.Button("↻", scale=0)
    status = gr.Markdown("")

    prefix = gr.Textbox(label="プレフィックス（プリセットから自動）", lines=1)
    prompt = gr.Textbox(label="プロンプト", lines=4, placeholder="1girl, ...")
    negative = gr.Textbox(label="ネガティブ", lines=2)

    with gr.Accordion("img2img（画像を置くと、その画像を下敷きに生成）", open=False):
        with gr.Row():
            in_image = gr.Image(type="pil", label="入力画像（空なら通常の txt2img）", height=280)
            strength = gr.Slider(
                0.1, 1.0, 0.6, step=0.05,
                label="変化の強さ（低いほど元画像に忠実、1.0 でほぼ無視）",
            )

    with gr.Row():
        aspect = gr.Radio(list(ASPECTS), value="縦 (プリセット)", label="サイズ")
        n = gr.Slider(1, 4, 1, step=1, label="枚数")

    with gr.Accordion("詳細設定", open=False):
        with gr.Row():
            steps = gr.Slider(10, 50, 28, step=1, label="Steps")
            cfg = gr.Slider(1, 10, 5, step=0.5, label="CFG")
        with gr.Row():
            sampler = gr.Dropdown(list(SAMPLERS), value="euler_a", label="Sampler")
            clip_skip = gr.Slider(1, 3, 2, step=1, label="Clip Skip")
            seed = gr.Number(-1, label="Seed (-1 でランダム)", precision=0)

    go = gr.Button("生成", variant="primary")
    gallery = gr.Gallery(label="結果", columns=2, height="auto")
    info = gr.Textbox(label="生成情報", interactive=False)

    preset_outputs = [status, prefix, negative, steps, cfg, sampler, clip_skip]
    model_dd.change(on_model_change, model_dd, preset_outputs)
    refresh.click(lambda: gr.update(choices=list_checkpoints()), None, model_dd)
    go.click(
        on_generate,
        [model_dd, prefix, prompt, negative, aspect, n, steps, cfg, sampler, clip_skip, seed,
         in_image, strength],
        [gallery, info],
    )
    if ckpts:
        demo.load(on_model_change, model_dd, preset_outputs)

demo.launch(inbrowser=True)
