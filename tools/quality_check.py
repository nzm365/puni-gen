"""品質回帰チェック: 固定 seed × 固定プロンプトで生成し、変更前後を比較する。

高速化の変更が絵を変えていないことを、目視ではなく数値で確認するためのツール。

使い方 (プロジェクトルートで実行):
  変更前の状態で:  .venv\\Scripts\\python.exe tools\\quality_check.py baseline
  変更後の状態で:  .venv\\Scripts\\python.exe tools\\quality_check.py after --compare baseline

知覚差 (LPIPS) も見たい場合は一度だけ:
  .venv\\Scripts\\python.exe -m pip install lpips

読み方の目安:
  - max/mean は画素差 (0〜1)。ビット一致なら両方 0
  - LPIPS < 0.05 なら実質同一、> 0.1 は別の絵と考える
  - NaN 画素が 1 つでもあれば、その変更は不合格 (fp16 VAE の破綻)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "tools" / "quality_runs"

# 破綻が出やすい構図を意図的に混ぜてある。dark は fp16 VAE の破綻が最初に出る場所
PROMPTS = {
    "girl": "1girl, solo, looking at viewer, upper body",
    "scene": "scenery, no humans, forest, river, sunlight",
    "duo": "2girls, sitting, cafe, smile",
    "hands": "1girl, waving, open hand, hand focus",
    "dark": "1girl, night, dark, low light, silhouette",
}
SEEDS = [1, 2, 3, 4]
SIZE = (832, 1216)
STEPS = 20


def generate(tag: str) -> Path:
    from engine import Engine, list_checkpoints

    ckpts = list_checkpoints()
    if not ckpts:
        sys.exit("models/checkpoints にモデルがありません")
    eng = Engine()
    print(eng.load(ckpts[0]))
    out = RUNS / tag
    out.mkdir(parents=True, exist_ok=True)
    times: dict[str, float] = {}
    for name, prompt in PROMPTS.items():
        for seed in SEEDS:
            key = f"{name}_{seed}"
            t0 = time.perf_counter()
            images, _, perf = eng.generate(
                "masterpiece, best quality, " + prompt,
                "worst quality, low quality",
                SIZE[0], SIZE[1], STEPS, 5.0, "euler_a", 2, seed, 1,
            )
            times[key] = round(time.perf_counter() - t0, 2)
            images[0].save(out / f"{key}.png")
            print(f"{key}: {times[key]}s  [{perf}]")
    (out / "times.json").write_text(json.dumps(times, indent=1), encoding="utf-8")
    total = sum(times.values())
    print(f"\n{tag}: {len(times)} 枚 / 合計 {total:.1f}s → {out}")
    return out


def compare(tag: str, base: str):
    a_dir, b_dir = RUNS / base, RUNS / tag
    lpips_fn = None
    try:
        import lpips  # type: ignore
        import torch

        net = lpips.LPIPS(net="alex", verbose=False)

        def lpips_fn(a, b):  # noqa: F811
            ta = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
            tb = torch.from_numpy(b).permute(2, 0, 1)[None] * 2 - 1
            return float(net(ta.float(), tb.float()))
    except ImportError:
        print("(lpips 未導入のため画素差のみ。pip install lpips で知覚差も出ます)\n")

    worst = ("", 0.0)
    print(f"{'ペア':<12}{'max':>7}{'mean':>9}{'NaN':>5}" + ("{:>8}".format("LPIPS") if lpips_fn else ""))
    for png in sorted(a_dir.glob("*.png")):
        pair = b_dir / png.name
        if not pair.exists():
            print(f"{png.stem:<12} (比較先なし)")
            continue
        a = np.asarray(Image.open(png), dtype=np.float32) / 255
        b = np.asarray(Image.open(pair), dtype=np.float32) / 255
        diff = np.abs(a - b)
        nan = int(np.isnan(a).sum() + np.isnan(b).sum())
        line = f"{png.stem:<12}{diff.max():>7.3f}{diff.mean():>9.4f}{nan:>5}"
        if lpips_fn:
            d = lpips_fn(a, b)
            line += f"{d:>8.3f}"
            if d > worst[1]:
                worst = (png.stem, d)
        print(line)

    ta = json.loads((a_dir / "times.json").read_text(encoding="utf-8"))
    tb = json.loads((b_dir / "times.json").read_text(encoding="utf-8"))
    sa, sb = sum(ta.values()), sum(tb.values())
    print(f"\n速度: {base} {sa:.1f}s → {tag} {sb:.1f}s ({sa / sb:.2f}x)")
    if worst[0]:
        print(f"最大 LPIPS: {worst[0]} = {worst[1]:.3f} (0.05 未満なら実質同一 / 0.1 超は別の絵)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", help="この実行につける名前 (例: baseline, after)")
    ap.add_argument("--compare", metavar="BASE", help="生成後にこのタグと比較する")
    args = ap.parse_args()
    generate(args.tag)
    if args.compare:
        print()
        compare(args.tag, args.compare)
