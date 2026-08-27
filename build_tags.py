"""Danbooru の公開 API からタグ辞書を取得して data/danbooru_tags.csv を作る。

プロンプト補完に使う静的データを生成するだけのツール。生成物をリポジトリに
同梱しておけば、実行時はオフラインで補完が効く。タグ辞書を更新したいとき
（新しいキャラクター名などを取り込みたいとき）にだけ再実行すればよい。

    .venv\\Scripts\\python.exe build_tags.py [取得件数]

既定で投稿数の多い上位 20000 タグを取得する。
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import requests

OUT = Path(__file__).parent / "data" / "danbooru_tags.csv"
API = "https://danbooru.donmai.us/tags.json"
UA = {"User-Agent": "PuniGen/1.0 (prompt tag autocomplete)"}

# meta タグ（category 5: highres, commentary_request 等）はプロンプトに使わないので除外。
# イラストレーター名 (category 1) も除外する: 特定の作家の画風を名指しで模倣する用途に
# 直結するため、補完候補としては出さない方針。
# 0=general 3=copyright 4=character を残す
KEEP_CATEGORIES = {0, 3, 4}
PER_PAGE = 1000


def fetch(target: int) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    page = 1
    while len(rows) < target:
        r = requests.get(
            API,
            params={
                "search[order]": "count",
                "search[is_deprecated]": "false",
                "limit": PER_PAGE,
                "page": page,
            },
            headers=UA,
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for t in batch:
            if t.get("category") not in KEEP_CATEGORIES:
                continue
            name = t.get("name") or ""
            count = t.get("post_count") or 0
            if name and count > 0:
                rows.append((name, t["category"], count))
        print(f"  page {page}: 累計 {len(rows)} タグ", flush=True)
        page += 1
        time.sleep(0.5)  # API に優しく
    return rows[:target]


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    print(f"Danbooru から上位 {target} タグを取得します...")
    rows = fetch(target)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)  # name, category, count
    size_kb = OUT.stat().st_size / 1024
    print(f"完了: {len(rows)} タグを {OUT} に保存しました（{size_kb:.0f} KB）。")


if __name__ == "__main__":
    main()
