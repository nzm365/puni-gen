"""最小構成の Gradio UI。モデルを切り替えるとプリセット（推奨設定）が自動で入る。"""
import queue
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import gradio as gr

import civitai
import prompt_autocomplete
from engine import Engine, InsufficientVram, SAMPLERS, fit_size, list_checkpoints, load_preset

OUT_DIR = Path(__file__).parent / "outputs"
EMB_DIR = Path(__file__).parent / "models" / "embeddings"
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
    # ユーザーがプロンプトを打っている間に、裏で 1 step 回して初回のもたつきを消す
    threading.Thread(target=engine.warmup, args=tuple(p["size"]), daemon=True).start()
    return (
        msg,
        p["prefix_pos"],
        p["default_neg"],
        p["steps"],
        p["cfg"],
        p["sampler"],
        p["clip_skip"],
    )


# 同一設定・同一 seed の再実行は定義上まったく同じ画像なので、再計算せず返す。
# 1 件 ≒ 数 MB × 最大 4 枚。8 件までに抑える
RESULT_CACHE: OrderedDict[tuple, tuple] = OrderedDict()


def _save_async(images, paths):
    """PNG 書き込みは表示をブロックしない。GPU はこの時点で既に空いている。

    compress_level 6→1 でエンコードが 2〜3 倍速くなる（ファイルは 1〜2 割大きくなる）。
    """
    def run():
        for im, p in zip(images, paths):
            im.save(p, compress_level=1)
    threading.Thread(target=run, daemon=True).start()


def on_generate(ckpt, prefix, prompt, negative, aspect, n, steps, cfg, sampler, clip_skip, seed,
                in_image, strength):
    if engine.current != ckpt:
        load_or_alert(ckpt)
    size = ASPECTS[aspect] or tuple(load_preset(ckpt)["size"])
    if in_image is not None:
        # i2i では入力のアスペクト比を保ち、選択サイズは画素数の目安として使う
        size = fit_size(in_image.size, size[0] * size[1])
    full_prompt = (prefix or "") + prompt
    seed, n, steps = int(seed), int(n), int(steps)

    # 明示 seed の再実行（生成情報の seed を入れて再現する手順）はキャッシュから即答
    cache_key = None
    if seed >= 0 and in_image is None:
        cache_key = (ckpt, full_prompt, negative, size, steps, float(cfg),
                     sampler, int(clip_skip), seed, n)
        hit = RESULT_CACHE.get(cache_key)
        if hit is not None:
            yield hit[0], hit[1] + "\n(同一設定・同一 seed のため再計算なし)"
            return

    # 生成はワーカースレッドで回し、ここでは途中経過を受け取って画面に流す
    q: queue.Queue = queue.Queue()
    res = {}

    def worker():
        try:
            res["out"] = engine.generate(
                full_prompt, negative, size[0], size[1],
                steps, float(cfg), sampler, int(clip_skip), seed, n,
                image=in_image, strength=float(strength),
                preview_cb=lambda s, total, imgs: q.put((s, total, imgs)),
            )
        except Exception as e:  # noqa: BLE001
            res["err"] = e
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while (msg := q.get()) is not None:
        s, total, imgs = msg
        if imgs is not None:
            yield imgs, f"生成中... {s}/{total} step（途中プレビュー）"
        else:
            yield gr.update(), f"生成中... {s}/{total} step"
    if "err" in res:
        e = res["err"]
        raise e if isinstance(e, gr.Error) else gr.Error(str(e))

    images, used_seed, perf = res["out"]
    if images is None:  # 中止された
        yield gr.update(), "中止しました（画像は保存されません）"
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = [OUT_DIR / f"{stamp}_{used_seed + i}.png" for i in range(len(images))]
    _save_async(images, paths)  # 保存完了を待たずに表示する
    mode = f"i2i (strength {strength})" if in_image is not None else "t2i"
    info = (
        f"seed: {used_seed}  size: {size[0]}x{size[1]}  mode: {mode}  prompt: {full_prompt}\n"
        f"[{perf}]"
    )
    if cache_key is not None:
        RESULT_CACHE[cache_key] = (images, info)
        while len(RESULT_CACHE) > 8:
            RESULT_CACHE.popitem(last=False)
    yield images, info


# ---------- モデルを追加（Civitai） ----------
def on_search(query):
    """検索してギャラリー用の (画像, キャプション) 一覧を返す。"""
    try:
        cands = civitai.search(query)
    except civitai.CivitaiError as e:
        return [], [], f"⚠ {e}", None
    if not cands:
        return [], [], "該当するモデルがありませんでした。別の語で試してください。", None
    thumbs = [
        (c.thumb_path or civitai.placeholder(), f"{c.label}\n{c.size_gb:.1f}GB")
        for c in cands
    ]
    return cands, thumbs, f"{len(cands)} 件見つかりました。選んでからダウンロードしてください。", None


def on_select(cands, evt: gr.SelectData):
    """ギャラリーで選ばれたモデルの詳細を出す。"""
    c = cands[evt.index]
    note = "（既に models/checkpoints にあります）" if c.exists else ""
    return evt.index, (
        f"**{c.model_name}** — {c.version_name} {note}\n\n"
        f"ベース: `{c.base_model}` / 作者: {c.creator} / "
        f"サイズ: {c.size_gb:.2f}GB / DL数: {c.downloads:,}\n\n"
        f"保存名: `{c.file_name}`"
    )


def on_download(cands, idx, progress=gr.Progress()):
    """選択中のモデルを落とし、モデル一覧を更新する。"""
    if idx is None:
        return "先に一覧からモデルを選んでください。", gr.update()
    try:
        msg = civitai.download(
            cands[idx],
            on_progress=lambda frac, text: progress(frac, desc=f"ダウンロード中 {text}"),
        )
    except civitai.CivitaiError as e:
        return f"⚠ {e}", gr.update()
    return msg, gr.update(choices=list_checkpoints())


emb_tokens = sorted(p.stem for p in EMB_DIR.glob("*.safetensors"))
with gr.Blocks(title="RTX Easy Image Gen") as demo:
    ckpts = list_checkpoints()
    gr.Markdown("## RTX Easy Image Gen")

    with gr.Tab("生成"):
        # 左に操作系、右に生成結果。縦に伸ばさず 1 画面に収める
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                with gr.Row():
                    model_dd = gr.Dropdown(
                        ckpts, value=ckpts[0] if ckpts else None, label="モデル", scale=4
                    )
                    refresh = gr.Button("↻", scale=0)
                status = gr.Markdown("")

                prefix = gr.Textbox(label="プレフィックス（プリセットから自動）", lines=1)
                prompt = gr.Textbox(
                    label="プロンプト", lines=4, placeholder="1girl, ...", elem_id="prompt_box"
                )
                negative = gr.Textbox(label="ネガティブ", lines=2, elem_id="neg_box")

                with gr.Row():
                    aspect = gr.Radio(list(ASPECTS), value="縦 (プリセット)", label="サイズ")
                    n = gr.Slider(1, 4, 1, step=1, label="枚数")

                with gr.Row():
                    go = gr.Button("生成", variant="primary", scale=3)
                    stop = gr.Button("中止", variant="stop", scale=1)

                with gr.Accordion("img2img（画像を置くと、その画像を下敷きに生成）", open=False):
                    in_image = gr.Image(
                        type="pil", label="入力画像（空なら通常の txt2img）", height=280
                    )
                    strength = gr.Slider(
                        0.1, 1.0, 0.6, step=0.05,
                        label="変化の強さ（低いほど元画像に忠実、1.0 でほぼ無視）",
                    )

                with gr.Accordion("詳細設定", open=False):
                    with gr.Row():
                        steps = gr.Slider(10, 50, 28, step=1, label="Steps")
                        cfg = gr.Slider(1, 10, 5, step=0.5, label="CFG")
                    with gr.Row():
                        sampler = gr.Dropdown(list(SAMPLERS), value="euler_a", label="Sampler")
                        clip_skip = gr.Slider(1, 3, 2, step=1, label="Clip Skip")
                        seed = gr.Number(-1, label="Seed (-1 でランダム)", precision=0)

            with gr.Column(scale=1):
                gallery = gr.Gallery(label="結果", columns=2, height=760)
                info = gr.Textbox(label="生成情報", interactive=False, lines=2)

    with gr.Tab("モデルを追加"):
        gr.Markdown(
            "Civitai から SDXL / Illustrious 系のチェックポイントを検索して追加します。"
            "ダウンロードには Civitai の API キーが必要です。"
        )
        with gr.Accordion("API キー設定", open=not civitai.load_token()):
            gr.Markdown(
                "[civitai.com](https://civitai.com) にログイン → 右上のアイコン → "
                "**Account settings** → **API Keys** で作成したキーを貼り付けてください。"
                "`config.local.json` に保存され、git 管理外です。"
            )
            with gr.Row():
                token_box = gr.Textbox(
                    value=civitai.load_token(), label="API キー", type="password", scale=4
                )
                token_save = gr.Button("保存", scale=0)
            token_status = gr.Markdown("")

        with gr.Row():
            query = gr.Textbox(
                label="検索", placeholder="illustrious, anime, pony ...",
                scale=4, submit_btn=True,
            )
            search_btn = gr.Button("検索", variant="primary", scale=0)

        results = gr.Gallery(
            label="検索結果（クリックで選択）", columns=5, height=420,
            object_fit="cover", preview=False,
        )
        detail = gr.Markdown("")
        dl_btn = gr.Button("選択したモデルをダウンロード", variant="primary")
        dl_status = gr.Markdown("")

        cands_state = gr.State([])
        sel_state = gr.State(None)

    preset_outputs = [status, prefix, negative, steps, cfg, sampler, clip_skip]
    model_dd.change(on_model_change, model_dd, preset_outputs)
    refresh.click(lambda: gr.update(choices=list_checkpoints()), None, model_dd)

    token_save.click(civitai.save_token, token_box, token_status)
    search_out = [cands_state, results, dl_status, sel_state]
    search_btn.click(on_search, query, search_out)
    query.submit(on_search, query, search_out)
    results.select(on_select, cands_state, [sel_state, detail])
    dl_btn.click(on_download, [cands_state, sel_state], [dl_status, model_dd])
    go.click(
        on_generate,
        [model_dd, prefix, prompt, negative, aspect, n, steps, cfg, sampler, clip_skip, seed,
         in_image, strength],
        [gallery, info],
    )
    # 中止は別イベントとして並走し、次の step で生成ループを打ち切る
    stop.click(engine.request_cancel, None, None)
    if ckpts:
        demo.load(on_model_change, model_dd, preset_outputs)

demo.launch(inbrowser=True, head=prompt_autocomplete.build_head(emb_tokens))
