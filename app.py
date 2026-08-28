"""最小構成の Gradio UI。モデルを切り替えるとプリセット（推奨設定）が自動で入る。"""
import html
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
import ui_style
import upscaler
from engine import (
    Engine, InsufficientVram, SAMPLERS, VARIATION_COUNT,
    fit_size, list_checkpoints, load_preset,
)

OUT_DIR = Path(__file__).parent / "outputs"
FAV_DIR = Path(__file__).parent / "favorites"
EMB_DIR = Path(__file__).parent / "models" / "embeddings"
engine = Engine()

# ---------- 進捗表示 ----------
# ギャラリー下の 1 行に出す HTML を組み立てる。「いま何をしているか」はここだけに出す。
# 待ちには 2 種類あり、見た目を分けている:
#   - step 数のように残りが数えられるもの  -> 実際の割合まで伸びるバー
#   - モデルの読み込みのように数えられないもの -> 丸い塊が往復するバー（止まっていない印）
def _bar(text: str, frac: float | None = None) -> str:
    """進捗バー。frac が None なら割合不明として塊を往復させる。"""
    body = html.escape(text)
    if frac is None:
        track = '<div class="pg-track pg-ind"><div class="pg-fill"></div></div>'
    else:
        pct = max(0.0, min(1.0, frac)) * 100
        track = f'<div class="pg-track"><div class="pg-fill" style="width:{pct:.1f}%"></div></div>'
    return f'<div class="pg"><div class="pg-text">{body}</div>{track}</div>'


def _done(text: str) -> str:
    """完了の表示。バーを満杯のまま残す。

    終わった瞬間に消すと、待たされていた人には「結局どうなったのか」が
    残らない。次の操作で置き換わるまで、終わったことを見えるままにする。
    """
    return (f'<div class="pg pg-done"><div class="pg-text">{html.escape(text)}</div>'
            '<div class="pg-track"><div class="pg-fill" style="width:100%"></div></div></div>')


def _note(text: str) -> str:
    """バーを出さない一言（中止しました、先に生成してください、など）。"""
    return f'<div class="pg pg-note"><div class="pg-text">{html.escape(text)}</div></div>'


ASPECTS = {
    "縦 (プリセット)": None,
    "縦 832x1216": (832, 1216),
    "縦 1024x1536": (1024, 1536),
    "横 1216x832": (1216, 832),
    "横 1536x1024": (1536, 1024),
    "正方 1024x1024": (1024, 1024),
}


def load_or_alert(ckpt, on_phase=None) -> str:
    """VRAM 不足はトレースバックではなく画面のエラー表示にする。"""
    try:
        return engine.load(ckpt, on_phase=on_phase)
    except InsufficientVram as e:
        raise gr.Error(str(e)) from e


def on_model_load(ckpt):
    """モデルを読み込み、進捗バーだけを更新する。

    ロードは 6GB 級のファイル読み込みで数十秒かかる。以前はここが同期呼び出しで、
    終わるまで画面に何も出ず「固まった」と見えていた。ワーカーに逃がして、
    engine から届く段階表示を progress に流す。

    出力を progress だけに絞っているのは、Gradio がジェネレータの
    実行中「出力として宣言した全コンポーネント」の枠を点滅させるため
    (statustracker の .generating: 2px の枠 + 2 秒周期の明滅)。
    プリセット欄まで出力に含めると、ロードしている数十秒のあいだ
    プレフィックスや Steps といった無関係な入力欄まで点滅してしまう。
    確定後の反映は .success で繋ぐ apply_preset（通常の関数）に任せる。
    """
    if not ckpt:
        yield ""
        return

    q: queue.Queue = queue.Queue()
    res = {}

    def worker():
        try:
            res["msg"] = load_or_alert(ckpt, on_phase=q.put)
        except Exception as e:  # noqa: BLE001
            res["err"] = e
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    yield _bar("モデルを準備しています...")
    while (text := q.get()) is not None:
        yield _bar(text)
    if "err" in res:
        e = res["err"]
        raise e if isinstance(e, gr.Error) else gr.Error(str(e))
    # VRAM の判定結果は画面には出さない。切り分けに要るのでコンソールには残す
    print(res["msg"])
    yield _done("読み込み完了")


def apply_preset(ckpt):
    """ロード完了後にプリセットを入れる。

    通常の関数なので、Gradio はこれらの欄を「生成中」として扱わない（枠が点滅しない）。
    """
    if not ckpt:
        return (gr.skip(),) * 6
    p = load_preset(ckpt)
    # ユーザーがプロンプトを打っている間に、裏で 1 step 回して初回のもたつきを消す
    threading.Thread(target=engine.warmup, args=tuple(p["size"]), daemon=True).start()
    return (
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
META_KEY = "puni_gen"
# 改名前 (RTX Easy Image Gen) に生成した画像が持つキー。読むときだけ見る。
# 落とすと、手元にある既存の画像から「少しだけ変える」「解像度を2倍」が
# できなくなる (生成条件を読み出せないため)
LEGACY_META_KEY = "rtx_easy_image_gen"


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
            raw = im.info.get(META_KEY) or im.info.get(LEGACY_META_KEY)
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
            yield (hit[0], _done("生成完了"), hit[1] + "\n(同一設定・同一 seed のため再計算なし)", hit[2])
            return

    # 生成はワーカースレッドで回し、ここでは途中経過を受け取って画面に流す
    q: queue.Queue = queue.Queue()
    res = {}

    def worker():
        try:
            # モデルのロードもここで行う。呼び出し側でやると最初の yield まで
            # 到達せず、その数十秒のあいだ画面が完全に無反応になる
            if engine.current != ckpt:
                load_or_alert(ckpt, on_phase=lambda t: q.put(("phase", t)))
            res["out"] = engine.generate(
                full_prompt, negative, size[0], size[1],
                steps, float(cfg), sampler, int(clip_skip), seed, n,
                image=in_image, strength=float(strength),
                preview_cb=lambda s, total, imgs: q.put(("step", s, total, imgs)),
                on_phase=lambda t: q.put(("phase", t)),
            )
        except Exception as e:  # noqa: BLE001
            res["err"] = e
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    # 前回の結果を残したままだと「生成中」の表示と一緒に前の絵が見え、
    # それが今まさに生成中の絵だと誤解される。開始時点で必ず空にする
    yield gr.update(value=[], selected_index=None), _bar("準備しています..."), gr.skip(), gr.skip()
    while (msg := q.get()) is not None:
        if msg[0] == "phase":
            yield gr.update(), _bar(msg[1]), gr.skip(), gr.skip()
            continue
        _, s, total, imgs = msg
        if imgs is not None:
            yield imgs, _bar(f"生成中... {s}/{total} step（途中プレビュー）", s / total), gr.skip(), gr.skip()
        else:
            yield gr.update(), _bar(f"生成中... {s}/{total} step", s / total), gr.skip(), gr.skip()
    if "err" in res:
        e = res["err"]
        raise e if isinstance(e, gr.Error) else gr.Error(str(e))

    images, used_seed, perf = res["out"]
    if images is None:  # 中止された
        yield gr.update(), _note("中止しました（画像は保存されません）"), gr.skip(), gr.skip()
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
    yield gr.update(value=images, selected_index=None), _done("生成完了"), info, last


# ---------- 高解像度化 (Hires.fix) ----------
def on_pick_result(evt: gr.SelectData):
    """結果ギャラリーで選ばれた画像の番号を覚える。"""
    return evt.index


def on_upscale(last, idx):
    """選択中の画像を拡大する。

    Real-ESRGAN を 1 回通すだけで、描き込みは足さない。
    以前は img2img で描き直す方式 (Hires.fix) だったが、絵柄が動くため置き換えた。
    engine.upscale 側の実装は残してあるので、戻したくなれば差し替えられる。
    """
    if not last or not last.get("images"):
        yield gr.update(), _note("先に画像を生成してください。"), gr.skip(), gr.skip()
        return
    images = last["images"]
    i = 0 if idx is None else min(int(idx), len(images) - 1)
    src = images[i]
    seeds = last.get("seeds") or []
    seed = seeds[i] if i < len(seeds) else last.get("seed", 0)

    yield gr.update(value=[], selected_index=None), _bar("準備しています..."), gr.skip(), gr.skip()

    q: queue.Queue = queue.Queue()
    res = {}

    def worker():
        try:
            # 初回だけ 17MB のダウンロードとモデル読み込みが入る。
            # そこも「拡大しています」の一言で覆うと、無反応の時間として見える
            res["out"] = upscaler.upscale(src, upscaler.SCALE, on_phase=q.put)
        except Exception as e:  # noqa: BLE001
            res["err"] = e
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while (text := q.get()) is not None:
        yield gr.update(), _bar(text), gr.skip(), gr.skip()

    if "err" in res:
        e = res["err"]
        raise gr.Error(str(e))

    out = [res["out"]]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"{stamp}_{seed}_x{upscaler.SCALE:g}.png"
    # 元の生成条件を引き継いで書く。拡大後も履歴からたどれるようにするため
    _save_async(out, [path], {
        "ckpt": last.get("ckpt"), "prompt": last.get("prompt", ""),
        "negative": last.get("negative", ""), "cfg": last.get("cfg"),
        "sampler": last.get("sampler"), "clip_skip": last.get("clip_skip"),
        "steps": last.get("steps"), "size": list(out[0].size), "seed": seed,
    })
    w, h = out[0].size
    info = (
        f"拡大: {src.width}x{src.height} -> {w}x{h}  seed: {seed}\n"
        f"file: {path.name}"
    )
    nxt = {**last, "images": out, "paths": [str(path)], "seeds": [seed]}
    yield gr.update(value=out, selected_index=None), _done("拡大完了"), info, nxt


# ---------- お気に入り ----------
# 星はフォントに任せると字形が環境ごとに変わって見栄えがしない。
# Gradio のボタンはラベルの HTML をエスケープするので、SVG を直接は埋め込めない。
# そこで elem_classes で状態を伝え、実際の星は CSS の背景画像として描く
# （ui_style.py 側で定義）。ラベル自体は空にしておく。
FAV_CLASS = "fav-btn"          # 未登録
FAV_CLASS_DONE = "fav-btn on"  # 登録済み


def _fav_name(state, idx) -> str | None:
    """選択中の画像のファイル名を返す（お気に入りの判定と移動に使う）。"""
    if not state or not state.get("paths"):
        return None
    i = 0 if idx is None else min(int(idx), len(state["paths"]) - 1)
    return Path(state["paths"][i]).name


def _fav_button(state, idx):
    """選択中の画像がお気に入り済みかどうかで、ボタンの見た目を返す。

    済みのときは primary（塗りつぶし）にして、ラベルにも印を付ける。
    色だけだと分かりにくい環境があるので、両方で示す。
    押すと解除できるので、済みでも押せる状態のままにする。
    """
    name = _fav_name(state, idx)
    on = bool(name and (FAV_DIR / name).exists())
    return gr.update(
        variant="primary" if on else "secondary",
        elem_classes=FAV_CLASS_DONE if on else FAV_CLASS,
        interactive=True,
    )


def list_favorites():
    """お気に入りフォルダの画像を新しい順で返す。"""
    if not FAV_DIR.exists():
        return []
    files = sorted(FAV_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


def on_favorite(last, idx):
    """選択中の画像のお気に入り状態を切り替える。

    追加は outputs から favorites への移動、解除はその逆。
    元の場所へ戻すので、解除しても画像は消えず履歴に並ぶ。
    """
    if not last or not last.get("images"):
        return "先に画像を生成してください。", gr.update()
    images = last["images"]
    i = 0 if idx is None else min(int(idx), len(images) - 1)
    FAV_DIR.mkdir(exist_ok=True)

    name = Path(last["paths"][i]).name
    in_out = OUT_DIR / name
    in_fav = FAV_DIR / name

    # --- 解除 ---
    if in_fav.exists():
        if in_out.exists():
            # 同名が両方にある（手で戻した等）。favorites 側だけ消して整合させる
            in_fav.unlink()
        else:
            shutil.move(str(in_fav), str(in_out))
        # 成功したかは星の色で分かるので、メッセージは出さない
        return gr.skip(), gr.update(value=list_favorites())

    # --- 追加 ---
    # 保存は裏で走っているので、書き終わるまで少しだけ待つ
    for _ in range(20):
        if in_out.exists():
            break
        time.sleep(0.1)

    if in_out.exists():
        shutil.move(str(in_out), str(in_fav))
    else:  # 保存が間に合わない場合はメモリ上の画像から直接書く
        images[i].save(in_fav, compress_level=1)
    return gr.skip(), gr.update(value=list_favorites())


def on_refresh_favorites():
    files = list_favorites()
    return gr.update(value=files), f"{len(files)} 枚"


# ---------- この絵の続き（変分） ----------
def on_variations(last, idx):
    """選択中の画像の seed を保ったまま、少しだけ違う絵を 4 枚出す。"""
    if not last or not last.get("images"):
        yield gr.update(), _note("先に画像を生成してください。"), gr.skip(), gr.skip()
        return
    images = last["images"]
    i = 0 if idx is None else min(int(idx), len(images) - 1)
    base_seed = last["seeds"][i]
    w, h = last["size"]

    q: queue.Queue = queue.Queue()
    res = {}

    def worker():
        try:
            if engine.current != last["ckpt"]:
                load_or_alert(last["ckpt"], on_phase=lambda t: q.put(("phase", t)))
            res["out"] = engine.variations(
                last["prompt"], last["negative"], w, h, last["steps"], last["cfg"],
                last["sampler"], last["clip_skip"], base_seed,
                preview_cb=lambda s, total, imgs: q.put(("step", s, total, imgs)),
                on_phase=lambda t: q.put(("phase", t)),
            )
        except Exception as e:  # noqa: BLE001
            res["err"] = e
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    # 選択も解除する（前の絵を選んだままだと新しい結果と対応がずれる）
    yield gr.update(value=[], selected_index=None), _bar("準備しています..."), gr.skip(), gr.skip()
    while (msg := q.get()) is not None:
        if msg[0] == "phase":
            yield gr.update(), _bar(msg[1]), gr.skip(), gr.skip()
            continue
        _, s, total, imgs = msg
        if imgs is not None:
            yield imgs, _bar(f"変分を生成中... {s}/{total} step（途中プレビュー）", s / total), gr.skip(), gr.skip()
        else:
            yield gr.update(), _bar(f"変分を生成中... {s}/{total} step", s / total), gr.skip(), gr.skip()
    if "err" in res:
        e = res["err"]
        raise e if isinstance(e, gr.Error) else gr.Error(str(e))

    out, _, perf = res["out"]
    if out is None:
        yield gr.update(), _note("中止しました（画像は保存されません）"), gr.skip(), gr.skip()
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
    yield gr.update(value=out, selected_index=None), _done("完了"), info, nxt


# ---------- 生成履歴 ----------
def list_history():
    """outputs/ の画像を新しい順に返す。"""
    if not OUT_DIR.exists():
        return []
    files = sorted(OUT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


def on_refresh_history():
    """一覧を更新し、選択を解除する。

    第 4 の戻り値は生成情報欄。ここは生成結果の表示と共用なので、
    せっかく出た結果を消さないよう触らない（gr.skip）。
    """
    files = list_history()
    return gr.update(value=files), f"{len(files)} 枚", None, gr.skip()


def on_refresh_history_keep():
    """一覧だけ更新し、選択中の画像は保つ。

    お気に入りの追加・解除の直後に使う。通常の更新は選択を解除するが、
    それだと解除した直後にもう一度押せず、「履歴から画像を選んでください」と
    出てしまう（実際にそうなっていた）。
    """
    files = list_history()
    return gr.update(value=files), f"{len(files)} 枚"


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
            f"{path.name}  size: {img.width}x{img.height}\n"
            "生成条件が記録されていない画像です（この機能より前に作られたもの）。\n"
            "お気に入りには入れられますが、少しだけ変える・解像度を上げるは実行できません。"
        )

    state = {
        "images": [img], "paths": [str(path)], "seeds": [meta.get("seed", 0)],
        "ckpt": meta.get("ckpt"), "prompt": meta.get("prompt", ""),
        "negative": meta.get("negative", ""), "cfg": meta.get("cfg", 5.0),
        "sampler": meta.get("sampler", "euler_a"), "clip_skip": meta.get("clip_skip", 2),
        "steps": meta.get("steps", 28), "size": meta.get("size", list(img.size)),
    }
    size = meta.get("size") or list(img.size)
    # 生成直後に出る表示と語順を揃えておくと、見比べたときに読みやすい
    return state, (
        f"seed: {meta.get('seed')}  size: {size[0]}x{size[1]}  "
        f"steps: {meta.get('steps')}  cfg: {meta.get('cfg')}  "
        f"{meta.get('sampler')}  clip skip: {meta.get('clip_skip')}\n"
        f"model: {meta.get('ckpt')}  file: {path.name}\n"
        f"prompt: {meta.get('prompt', '')}"
    )


def _need_meta(state):
    """条件が無い画像で再生成系を押されたときの共通チェック。"""
    if not state:
        return "先に履歴から画像を選んでください。"
    if state.get("no_meta"):
        return ("この画像には生成条件が記録されていないため実行できません"
                "（今回の更新より前に作られた画像です）。新しく生成した画像なら使えます。")
    return None


# ---------- 操作対象の切り替え ----------
# ボタンは 1 組しか置かない。押されたとき、右カラムでどちらのサブタブを開いているかで
# 「今回の生成結果」と「履歴で選んだ画像」のどちらに作用するかを決める。
def _target(which, last, picked, hist_sel):
    """(状態, 番号, エラーメッセージ) を返す。"""
    if which == "history":
        if not hist_sel:
            return None, 0, "履歴から画像を選んでください。"
        if hist_sel.get("no_meta"):
            return None, 0, ("この画像には生成条件が記録されていないため実行できません"
                             "（この機能より前に作られた画像です）。")
        return hist_sel, 0, None
    if not last or not last.get("images"):
        return None, 0, "先に画像を生成してください。"
    return last, picked, None


def fav_button_state(which, last, picked, hist_sel):
    """いま操作対象になっている画像で、お気に入りボタンの見た目を決める。"""
    if which == "history":
        return _fav_button(hist_sel, 0)
    return _fav_button(last, picked)


def on_variations_target(which, last, picked, hist_sel):
    state, idx, msg = _target(which, last, picked, hist_sel)
    if msg:
        yield gr.update(), _note(msg), gr.skip(), gr.skip()
        return
    yield from on_variations(state, idx)


def on_upscale_target(which, last, picked, hist_sel):
    state, idx, msg = _target(which, last, picked, hist_sel)
    if msg:
        yield gr.update(), _note(msg), gr.skip(), gr.skip()
        return
    yield from on_upscale(state, idx)


def on_favorite_target(which, last, picked, hist_sel):
    # お気に入りは生成条件が要らないので no_meta でも通す
    if which == "history":
        if not hist_sel:
            return "履歴から画像を選んでください。", gr.update()
        return on_favorite(hist_sel, 0)
    return on_favorite(last, picked)


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
with gr.Blocks(title="PuniGen") as demo:
    ckpts = list_checkpoints()
    gr.Markdown("## PuniGen")

    with gr.Tabs() as tabs:
        with gr.Tab("生成", id="gen"):
            # 左に操作系、右に生成結果。縦に伸ばさず 1 画面に収める
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    # 選択欄自身の枠は残し、ボタンをその右へ密着させて
                    # 1 つの入力グループに見せる（CSS 側で指定）
                    with gr.Row(elem_id="model_row"):
                        model_dd = gr.Dropdown(
                            ckpts, value=ckpts[0] if ckpts else None, label="モデル", scale=4
                        )
                        # min_width を切らないと、Gradio の既定でボタンが横に広がる
                        refresh = gr.Button("↻", scale=0, min_width=0)
                    # 進行中のことと、終わったことだけを出す 1 行。
                    # 「いま何をしているか」の置き場はここ 1 箇所に限る。
                    # 完成した絵の情報は右の info が受け持つ
                    progress = gr.HTML("", elem_id="progress_line")

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
                    # 今回の生成結果と過去の履歴を、同じ場所でサブタブとして切り替える。
                    # 下のボタンは 1 組だけ置き、開いている側の画像に作用させる
                    with gr.Tabs() as out_tabs:
                        with gr.Tab("今回の結果", id="current") as cur_tab:
                            gallery = gr.Gallery(
                                label="結果", columns=2, height=680, object_fit="contain",
                            )
                        with gr.Tab("履歴", id="history") as hist_tab:
                            hist_count = gr.Markdown("")
                            hist_gallery = gr.Gallery(
                                label="生成履歴（クリックで選択）", columns=3, height=620,
                                object_fit="contain",
                            )

                    with gr.Row():
                        variation = gr.Button(
                            f"少しだけ変える（{VARIATION_COUNT} 枚）", variant="secondary", scale=3
                        )
                        hires = gr.Button(
                            f"解像度を {upscaler.SCALE:g} 倍", variant="secondary", scale=3
                        )
                        # お気に入り済みかどうかを色で示すので、状態に応じて variant を差し替える
                        favorite = gr.Button(
                            "", variant="secondary", scale=1,
                            elem_classes=FAV_CLASS,
                        )
                    info = gr.Textbox(label="生成情報", interactive=False, lines=3)

                    # 直前の生成条件（高解像度化で同じプロンプト・設定を再現するため）と
                    # 結果ギャラリーで選ばれている画像の番号
                    last_gen = gr.State(None)
                    picked = gr.State(None)
                    # 履歴側で選ばれている画像の条件と、いま開いているサブタブ
                    hist_sel = gr.State(None)
                    which_out = gr.State("current")

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

    # お気に入りボタンの表示更新は多くの操作の後に走るので、引数をまとめておく
    _fav_args = ([which_out, last_gen, picked, hist_sel], favorite)

    preset_outputs = [prefix, negative, steps, cfg, sampler, clip_skip]
    # ロード中の点滅を progress だけに閉じ込めるため、ロード（ジェネレータ）と
    # 反映（通常の関数）を 2 段に分ける。失敗したらプリセットは入れないので .success
    model_dd.change(on_model_load, model_dd, progress).success(
        apply_preset, model_dd, preset_outputs
    )
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
        [gallery, progress, info, last_gen],
    ).then(lambda: None, None, picked).then(fav_button_state, *_fav_args)
    gallery.select(on_pick_result, None, picked).then(
        fav_button_state, *_fav_args
    )
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
    # タブを開いたときに自動で読み直す（再読み込みボタンは置かない）。
    # 生成やお気に入り移動でフォルダの中身が変わるので、開くたびに最新にする
    # --- 右カラムのサブタブ ---
    # どちらを開いているかを覚え、ボタンはその側の画像に作用する
    # どちらのサブタブを開いているかを覚える。
    # Tabs の select が返す evt.value はラベル文字列なので、id と取り違えないよう
    # タブごとに定数を返す形にしている。
    cur_tab.select(lambda: "current", None, which_out).then(
        fav_button_state, [which_out, last_gen, picked, hist_sel], favorite
    )

    hist_tab.select(
        lambda: (*on_refresh_history(), "history"), None,
        [hist_gallery, hist_count, hist_sel, info, which_out],
    ).then(fav_button_state, *_fav_args)
    hist_gallery.select(on_pick_history, None, [hist_sel, info]).then(
        fav_button_state, *_fav_args
    )
    hist_gallery.preview_close(None, None, None, js=_EXIT_FS)

    # --- 操作ボタン（1 組で両方のサブタブを兼ねる）---
    # 再生成したら「今回の結果」へ切り替え、履歴一覧も更新する。
    # 結果の表示場所を 1 か所に保ちつつ、履歴に新しい画像を反映させるため。
    def _to_current():
        return gr.Tabs(selected="current"), "current"

    _refresh_hist = (on_refresh_history, None,
                     [hist_gallery, hist_count, hist_sel, info])

    variation.click(
        on_variations_target, [which_out, last_gen, picked, hist_sel],
        [gallery, progress, info, last_gen],
    ).then(_to_current, None, [out_tabs, which_out]).then(
        lambda: None, None, picked
    ).then(fav_button_state, *_fav_args).then(*_refresh_hist)

    hires.click(
        on_upscale_target, [which_out, last_gen, picked, hist_sel],
        [gallery, progress, info, last_gen],
    ).then(_to_current, None, [out_tabs, which_out]).then(
        lambda: None, None, picked
    ).then(fav_button_state, *_fav_args).then(*_refresh_hist)

    # お気に入りは画像が増えないのでサブタブは切り替えない
    # 追加・解除で outputs と favorites の間を移動するので一覧は更新するが、
    # 選択は保つ（続けて押し直せるようにするため）
    favorite.click(
        on_favorite_target, [which_out, last_gen, picked, hist_sel],
        [info, fav_gallery],
    ).then(fav_button_state, *_fav_args).then(
        on_refresh_history_keep, None, [hist_gallery, hist_count]
    )

    fav_tab.select(on_refresh_favorites, None, [fav_gallery, fav_count])
    demo.load(on_refresh_favorites, None, [fav_gallery, fav_count])
    # 中止は別イベントとして並走し、次の step で生成ループを打ち切る
    stop.click(engine.request_cancel, None, None)
    if ckpts:
        demo.load(on_model_load, model_dd, progress).success(
            apply_preset, model_dd, preset_outputs
        )

demo.launch(
    inbrowser=True,
    head=prompt_autocomplete.build_head(emb_tokens) + gallery_fix.build_head(),
    css=ui_style.build_css(),
    # Gradio は自分が作ったファイル以外の配信を 403 で拒む。検索結果のサムネイルは
    # civitai.py が .thumb_cache に自前で保存しているため、明示的に許可しないと
    # ギャラリーが空のまま表示される
    # 履歴タブは outputs/ の画像をパスで表示するので、ここも配信を許可する
    allowed_paths=[str(civitai.THUMB_DIR), str(FAV_DIR), str(OUT_DIR)],
)
