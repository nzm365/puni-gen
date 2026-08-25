"""最小構成の Gradio UI。モデルを切り替えるとプリセット（推奨設定）が自動で入る。"""
import json
import queue
import shutil
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image, PngImagePlugin

import civitai
import gallery_fix
import prompt_autocomplete
from engine import (
    Engine, HIRES_SCALE, InsufficientVram, SAMPLERS, VARIATION_COUNT,
    fit_size, list_checkpoints, load_preset,
)

OUT_DIR = Path(__file__).parent / "outputs"
FAV_DIR = Path(__file__).parent / "favorites"
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


# PNG に埋め込むキー。JSON 側が本体で、parameters は他ツール互換の読み物
META_KEY = "rtx_easy_image_gen"


def _png_meta(params: dict) -> PngImagePlugin.PngInfo:
    """生成条件を PNG のテキストチャンクに詰める。

    履歴タブから「少しだけ変える」「高解像度化」をするには、その絵を作ったときの
    プロンプトや seed が要る。画像自体に持たせておけば、ファイルを移動しても
    （お気に入りへ移しても）条件が失われない。
    """
    info = PngImagePlugin.PngInfo()
    info.add_text(META_KEY, json.dumps(params, ensure_ascii=False))
    # A1111 系ツールが読む慣習的な 1 行表記も添えておく
    info.add_text("parameters", (
        f"{params.get('prompt', '')}\n"
        f"Negative prompt: {params.get('negative', '')}\n"
        f"Steps: {params.get('steps')}, Sampler: {params.get('sampler')}, "
        f"CFG scale: {params.get('cfg')}, Seed: {params.get('seed')}, "
        f"Size: {params.get('size', [0, 0])[0]}x{params.get('size', [0, 0])[1]}, "
        f"Clip skip: {params.get('clip_skip')}, Model: {params.get('ckpt', '')}"
    ))
    return info


def read_meta(path) -> dict | None:
    """PNG から生成条件を読み戻す。無ければ None（古い画像や外部の画像）。"""
    try:
        with Image.open(path) as im:
            raw = im.info.get(META_KEY)
        return json.loads(raw) if raw else None
    except (OSError, ValueError):
        return None


def _save_async(images, paths, params: dict | None = None):
    """PNG 書き込みは表示をブロックしない。GPU はこの時点で既に空いている。

    compress_level 6→1 でエンコードが 2〜3 倍速くなる（ファイルは 1〜2 割大きくなる）。
    """
    def run():
        for i, (im, p) in enumerate(zip(images, paths)):
            meta = None
            if params is not None:
                one = dict(params)
                seeds = params.get("seeds")
                if seeds and i < len(seeds):  # 画像ごとに seed は違う
                    one["seed"] = seeds[i]
                one.pop("seeds", None)
                meta = _png_meta(one)
            im.save(p, compress_level=1, pnginfo=meta)
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
            yield (hit[0], hit[1] + "\n(同一設定・同一 seed のため再計算なし)", hit[2])
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
    # 前回の結果を残したままだと「生成中」の表示と一緒に前の絵が見え、
    # それが今まさに生成中の絵だと誤解される。開始時点で必ず空にする
    yield gr.update(value=[], selected_index=None), "生成中... 0 step", gr.skip()
    while (msg := q.get()) is not None:
        s, total, imgs = msg
        if imgs is not None:
            yield imgs, f"生成中... {s}/{total} step（途中プレビュー）", gr.skip()
        else:
            yield gr.update(), f"生成中... {s}/{total} step", gr.skip()
    if "err" in res:
        e = res["err"]
        raise e if isinstance(e, gr.Error) else gr.Error(str(e))

    images, used_seed, perf = res["out"]
    if images is None:  # 中止された
        yield gr.update(), "中止しました（画像は保存されません）", gr.skip()
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = [OUT_DIR / f"{stamp}_{used_seed + i}.png" for i in range(len(images))]
    meta = {
        "ckpt": ckpt, "prompt": full_prompt, "negative": negative, "cfg": float(cfg),
        "sampler": sampler, "clip_skip": int(clip_skip), "steps": steps,
        "size": list(size), "seed": used_seed,
        "seeds": [used_seed + i for i in range(len(images))],
    }
    _save_async(images, paths, meta)  # 保存完了を待たずに表示する
    mode = f"i2i (strength {strength})" if in_image is not None else "t2i"
    info = (
        f"seed: {used_seed}  size: {size[0]}x{size[1]}  mode: {mode}  prompt: {full_prompt}\n"
        f"[{perf}]"
    )
    # 高解像度化で同じ条件を再現するために生成条件を持ち回す
    last = {
        "images": images, "ckpt": ckpt, "prompt": full_prompt, "negative": negative,
        "cfg": float(cfg), "sampler": sampler, "clip_skip": int(clip_skip),
        "seed": used_seed, "size": size, "steps": steps,
        # 画像ごとの保存先と seed（お気に入り移動と変分で使う）
        "paths": [str(p) for p in paths],
        "seeds": [used_seed + i for i in range(len(images))],
    }
    if cache_key is not None:
        RESULT_CACHE[cache_key] = (images, info, last)
        while len(RESULT_CACHE) > 8:
            RESULT_CACHE.popitem(last=False)
    yield gr.update(value=images, selected_index=None), info, last


# ---------- 高解像度化 (Hires.fix) ----------
def on_pick_result(evt: gr.SelectData):
    """結果ギャラリーで選ばれた画像の番号を覚える。"""
    return evt.index


def on_upscale(last, idx):
    """選択中の生成結果を拡大し、img2img で細部を描き足す（倍率は engine.HIRES_SCALE）。"""
    if not last or not last.get("images"):
        yield gr.update(), "先に画像を生成してください。", gr.skip()
        return
    images = last["images"]
    i = 0 if idx is None else min(int(idx), len(images) - 1)
    src = images[i]

    if engine.current != last["ckpt"]:
        load_or_alert(last["ckpt"])

    q: queue.Queue = queue.Queue()
    res = {}

    def worker():
        try:
            res["out"] = engine.upscale(
                src, last["prompt"], last["negative"], last["cfg"], last["sampler"],
                last["clip_skip"], last["seed"],
                preview_cb=lambda s, total, imgs: q.put((s, total, imgs)),
            )
        except Exception as e:  # noqa: BLE001
            res["err"] = e
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    yield gr.update(value=[], selected_index=None), "高解像度化中... 0 step", gr.skip()
    while (msg := q.get()) is not None:
        s, total, imgs = msg
        if imgs is not None:
            yield imgs, f"高解像度化中... {s}/{total} step（途中プレビュー）", gr.skip()
        else:
            yield gr.update(), f"高解像度化中... {s}/{total} step", gr.skip()
    if "err" in res:
        e = res["err"]
        raise e if isinstance(e, gr.Error) else gr.Error(str(e))

    out, used_seed, perf = res["out"]
    if out is None:  # 中止された
        yield gr.update(), "中止しました（画像は保存されません）", gr.skip()
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"{stamp}_{used_seed}_hires.png"
    _save_async(out, [path], {
        "ckpt": last["ckpt"], "prompt": last["prompt"], "negative": last["negative"],
        "cfg": last["cfg"], "sampler": last["sampler"], "clip_skip": last["clip_skip"],
        "steps": last.get("steps"), "size": list(out[0].size), "seed": used_seed,
    })
    w, h = out[0].size
    info = (
        f"高解像度化: {src.width}x{src.height} -> {w}x{h}  seed: {used_seed}\n"
        f"[{perf}]"
    )
    # 拡大結果を起点にさらに高解像度化しないよう、状態は元のまま据え置く
    yield gr.update(value=out, selected_index=None), info, gr.skip()


# ---------- お気に入り ----------
def list_favorites():
    """お気に入りフォルダの画像を新しい順で返す。"""
    if not FAV_DIR.exists():
        return []
    files = sorted(FAV_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


def on_favorite(last, idx):
    """選択中の画像を outputs から favorites へ移す。"""
    if not last or not last.get("images"):
        return "先に画像を生成してください。", gr.update()
    images = last["images"]
    i = 0 if idx is None else min(int(idx), len(images) - 1)
    FAV_DIR.mkdir(exist_ok=True)

    src = Path(last["paths"][i])
    dst = FAV_DIR / src.name
    if dst.exists():
        return f"すでにお気に入りにあります: {dst.name}", gr.update()

    # 保存は裏で走っているので、書き終わるまで少しだけ待つ
    for _ in range(20):
        if src.exists():
            break
        time.sleep(0.1)

    if src.exists():
        shutil.move(str(src), str(dst))
    else:  # 保存が間に合わない場合はメモリ上の画像から直接書く
        images[i].save(dst, compress_level=1)
    return f"お気に入りに移動しました: {dst.name}", gr.update(value=list_favorites())


def on_refresh_favorites():
    files = list_favorites()
    return gr.update(value=files), f"{len(files)} 枚"


# ---------- この絵の続き（変分） ----------
def on_variations(last, idx):
    """選択中の画像の seed を保ったまま、少しだけ違う絵を 4 枚出す。"""
    if not last or not last.get("images"):
        yield gr.update(), "先に画像を生成してください。", gr.skip()
        return
    images = last["images"]
    i = 0 if idx is None else min(int(idx), len(images) - 1)
    base_seed = last["seeds"][i]
    w, h = last["size"]

    if engine.current != last["ckpt"]:
        load_or_alert(last["ckpt"])

    q: queue.Queue = queue.Queue()
    res = {}

    def worker():
        try:
            res["out"] = engine.variations(
                last["prompt"], last["negative"], w, h, last["steps"], last["cfg"],
                last["sampler"], last["clip_skip"], base_seed,
                preview_cb=lambda s, total, imgs: q.put((s, total, imgs)),
            )
        except Exception as e:  # noqa: BLE001
            res["err"] = e
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    # 選択も解除する（前の絵を選んだままだと新しい結果と対応がずれる）
    yield gr.update(value=[], selected_index=None), "変分を生成中... 0 step", gr.skip()
    while (msg := q.get()) is not None:
        s, total, imgs = msg
        if imgs is not None:
            yield imgs, f"変分を生成中... {s}/{total} step（途中プレビュー）", gr.skip()
        else:
            yield gr.update(), f"変分を生成中... {s}/{total} step", gr.skip()
    if "err" in res:
        e = res["err"]
        raise e if isinstance(e, gr.Error) else gr.Error(str(e))

    out, _, perf = res["out"]
    if out is None:
        yield gr.update(), "中止しました（画像は保存されません）", gr.skip()
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = [OUT_DIR / f"{stamp}_{base_seed}_var{k + 1}.png" for k in range(len(out))]
    _save_async(out, paths, {
        "ckpt": last["ckpt"], "prompt": last["prompt"], "negative": last["negative"],
        "cfg": last["cfg"], "sampler": last["sampler"], "clip_skip": last["clip_skip"],
        "steps": last["steps"], "size": [w, h], "seed": base_seed,
    })
    info = (
        f"少しだけ変える: seed {base_seed} を維持したまま {len(out)} 枚"
        f"（元画像: {i + 1} 枚目）\n[{perf}]"
    )
    # 出てきた変分をさらに起点にできるよう、状態を差し替える
    nxt = {
        **last, "images": out,
        "paths": [str(p) for p in paths],
        "seeds": [base_seed] * len(out),
    }
    yield gr.update(value=out, selected_index=None), info, nxt


# ---------- 生成履歴 ----------
def list_history():
    """outputs/ の画像を新しい順に返す。"""
    if not OUT_DIR.exists():
        return []
    files = sorted(OUT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


def on_refresh_history():
    files = list_history()
    return gr.update(value=files), f"{len(files)} 枚", None, "画像を選ぶと操作できます。"


def on_pick_history(evt: gr.SelectData):
    """履歴の画像を選ぶ。埋め込まれた生成条件を読み、そのまま再操作できる形にする。

    返す辞書は on_variations / on_upscale / on_favorite が受け取る形と同じなので、
    生成タブ用に書いたハンドラをそのまま使い回せる。
    """
    files = list_history()
    if evt.index is None or evt.index >= len(files):
        return None, "選択できませんでした。再読み込みしてください。"
    path = Path(files[evt.index])
    meta = read_meta(path)
    try:
        img = Image.open(path)
        img.load()
        img = img.convert("RGB")
    except OSError as e:
        return None, f"画像を開けませんでした: {e}"

    if meta is None:
        # 生成条件が無い画像（この機能より前に作られたもの、外部から置いたもの）。
        # お気に入りへの移動だけは条件が要らないので、それだけ可能にする。
        state = {
            "images": [img], "paths": [str(path)], "seeds": [0],
            "ckpt": None, "prompt": "", "negative": "", "cfg": 5.0,
            "sampler": "euler_a", "clip_skip": 2, "steps": 28,
            "size": list(img.size), "no_meta": True,
        }
        return state, (
            f"**{path.name}**（{img.width}x{img.height}）\n\n"
            "この画像には生成条件が記録されていません。今回の更新より前に作られた画像です。\n"
            "**お気に入りへの移動はできます**が、「少しだけ変える」「高解像度化」は"
            "元のプロンプトや seed が分からないため実行できません。"
        )

    state = {
        "images": [img], "paths": [str(path)], "seeds": [meta.get("seed", 0)],
        "ckpt": meta.get("ckpt"), "prompt": meta.get("prompt", ""),
        "negative": meta.get("negative", ""), "cfg": meta.get("cfg", 5.0),
        "sampler": meta.get("sampler", "euler_a"), "clip_skip": meta.get("clip_skip", 2),
        "steps": meta.get("steps", 28), "size": meta.get("size", list(img.size)),
    }
    return state, (
        f"**{path.name}**（{img.width}x{img.height}）\n\n"
        f"seed: `{meta.get('seed')}` / Steps: {meta.get('steps')} / "
        f"CFG: {meta.get('cfg')} / {meta.get('sampler')} / Clip Skip: {meta.get('clip_skip')}\n\n"
        f"モデル: `{meta.get('ckpt')}`\n\n"
        f"プロンプト: {meta.get('prompt', '')}"
    )


def _need_meta(state):
    """条件が無い画像で再生成系を押されたときの共通チェック。"""
    if not state:
        return "先に履歴から画像を選んでください。"
    if state.get("no_meta"):
        return ("この画像には生成条件が記録されていないため実行できません"
                "（今回の更新より前に作られた画像です）。新しく生成した画像なら使えます。")
    return None


def on_hist_variations(state, idx):
    msg = _need_meta(state)
    if msg:
        yield gr.update(), msg, gr.skip()
        return
    yield from on_variations(state, idx)


def on_hist_upscale(state, idx):
    msg = _need_meta(state)
    if msg:
        yield gr.update(), msg, gr.skip()
        return
    yield from on_upscale(state, idx)


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

    with gr.Tabs() as tabs:
        with gr.Tab("生成", id="gen"):
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
                    gallery = gr.Gallery(
                        label="結果", columns=2, height=760, object_fit="contain",
                    )
                    with gr.Row():
                        variation = gr.Button(
                            f"この絵を少しだけ変える（{VARIATION_COUNT} 枚）", variant="secondary", scale=2
                        )
                        favorite = gr.Button("★ お気に入り", variant="secondary", scale=1)
                    hires = gr.Button(
                        f"選択した画像を {HIRES_SCALE:g} 倍に高解像度化", variant="secondary"
                    )
                    info = gr.Textbox(label="生成情報", interactive=False, lines=2)

                    # 直前の生成条件（高解像度化で同じプロンプト・設定を再現するため）と
                    # 結果ギャラリーで選ばれている画像の番号
                    last_gen = gr.State(None)
                    picked = gr.State(None)

        with gr.Tab("生成履歴", id="history") as hist_tab:
            gr.Markdown(
                "`outputs/` に保存された画像を新しい順に表示します"
                "（このタブを開くたびに自動で最新になります）。"
                "画像を選ぶと、生成タブに戻らずにそのまま操作できます。"
            )
            hist_count = gr.Markdown("")

            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    hist_gallery = gr.Gallery(
                        label="生成履歴（クリックで選択）", columns=4, height=640,
                        object_fit="contain",
                    )
                with gr.Column(scale=1):
                    hist_detail = gr.Markdown("画像を選ぶと操作できます。")
                    with gr.Row():
                        hist_variation = gr.Button(
                            f"この絵を少しだけ変える（{VARIATION_COUNT} 枚）",
                            variant="secondary", scale=2,
                        )
                        hist_favorite = gr.Button("★ お気に入り", variant="secondary", scale=1)
                    hist_hires = gr.Button(
                        f"選択した画像を {HIRES_SCALE:g} 倍に高解像度化", variant="secondary"
                    )
                    gr.Markdown(
                        "「少しだけ変える」「高解像度化」を押すと**生成タブに移動**して"
                        "結果が表示されます。そのまま続けて操作できます。"
                    )
                    hist_info = gr.Textbox(label="情報", interactive=False, lines=2)

            # 選択中の画像とその生成条件（生成タブの last_gen と同じ形）
            hist_sel = gr.State(None)
            hist_idx = gr.State(0)

        with gr.Tab("お気に入り", id="favorites") as fav_tab:
            gr.Markdown(
                "生成タブで **★ お気に入り** を押した画像がここに集まります。"
                "ファイルは `outputs/` から `favorites/` へ移動するので、"
                "`outputs/` を整理しても消えません。"
            )
            fav_count = gr.Markdown("")
            fav_gallery = gr.Gallery(
                label="お気に入り", columns=4, height=720, object_fit="contain"
            )

        with gr.Tab("モデルを追加", id="models"):
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
        [gallery, info, last_gen],
    ).then(lambda: None, None, picked)
    gallery.select(on_pick_result, None, picked)
    # 拡大表示を閉じたらブラウザのフルスクリーンも解除する。
    # Gallery 側は fullscreen を解除しないため（frontend に exitFullscreen が無い）、
    # 閉じても全画面の格子表示が残り、元の画面に戻れなくなる。
    _EXIT_FS = (
        "() => { const b = document.querySelector('button[aria-label=\"Exit fullscreen mode\"]');"
        " if (b) b.click();"
        " if (document.fullscreenElement) document.exitFullscreen(); }"
    )
    gallery.preview_close(None, None, None, js=_EXIT_FS)
    fav_gallery.preview_close(None, None, None, js=_EXIT_FS)
    # 生成・変分・高解像度化のたびに選択番号を捨てる（結果が入れ替わるため）
    hires.click(on_upscale, [last_gen, picked], [gallery, info, last_gen]).then(
        lambda: None, None, picked
    )
    variation.click(on_variations, [last_gen, picked], [gallery, info, last_gen]).then(
        lambda: None, None, picked
    )
    favorite.click(on_favorite, [last_gen, picked], [info, fav_gallery])
    # タブを開いたときに自動で読み直す（再読み込みボタンは置かない）。
    # 生成やお気に入り移動でフォルダの中身が変わるので、開くたびに最新にする
    hist_tab.select(on_refresh_history, None,
                    [hist_gallery, hist_count, hist_sel, hist_detail])
    demo.load(on_refresh_history, None, [hist_gallery, hist_count, hist_sel, hist_detail])
    hist_gallery.select(on_pick_history, None, [hist_sel, hist_detail])
    hist_gallery.preview_close(None, None, None, js=_EXIT_FS)
    # 生成タブ用のハンドラをそのまま使う（state の形を合わせてある）。
    # 実行後は履歴が増えるので一覧も更新する
    # 履歴からの再生成は、結果を生成タブのギャラリーに出して画面ごと移動する。
    # 履歴タブに結果表示を二重に持たず、移動先でそのまま続けて操作できるようにするため。
    # last_gen も更新するので、生成タブ側のボタンがそのまま効く。
    def _to_gen():
        return gr.Tabs(selected="gen")

    hist_variation.click(
        on_hist_variations, [hist_sel, hist_idx], [gallery, info, last_gen]
    ).then(_to_gen, None, tabs).then(
        on_refresh_history, None, [hist_gallery, hist_count, hist_sel, hist_detail]
    )
    hist_hires.click(
        on_hist_upscale, [hist_sel, hist_idx], [gallery, info, last_gen]
    ).then(_to_gen, None, tabs).then(
        on_refresh_history, None, [hist_gallery, hist_count, hist_sel, hist_detail]
    )
    # お気に入りは画像が増えるわけではないので履歴タブに留まる（続けて選別しやすい）
    hist_favorite.click(
        on_favorite, [hist_sel, hist_idx], [hist_info, fav_gallery]
    ).then(on_refresh_history, None, [hist_gallery, hist_count, hist_sel, hist_detail])
    fav_tab.select(on_refresh_favorites, None, [fav_gallery, fav_count])
    demo.load(on_refresh_favorites, None, [fav_gallery, fav_count])
    # 中止は別イベントとして並走し、次の step で生成ループを打ち切る
    stop.click(engine.request_cancel, None, None)
    if ckpts:
        demo.load(on_model_change, model_dd, preset_outputs)

demo.launch(
    inbrowser=True,
    head=prompt_autocomplete.build_head(emb_tokens) + gallery_fix.build_head(),
    # Gradio は自分が作ったファイル以外の配信を 403 で拒む。検索結果のサムネイルは
    # civitai.py が .thumb_cache に自前で保存しているため、明示的に許可しないと
    # ギャラリーが空のまま表示される
    # 履歴タブは outputs/ の画像をパスで表示するので、ここも配信を許可する
    allowed_paths=[str(civitai.THUMB_DIR), str(FAV_DIR), str(OUT_DIR)],
)
