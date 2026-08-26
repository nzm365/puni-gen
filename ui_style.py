"""画面まわりの CSS（Gradio の launch(css=...) に渡す）。

やっていることは 2 つ。

1. 表示倍率を下げる
   Gradio の既定は余白も文字も大きめで、フル HD だとページ全体が 1070px になり
   縦スクロールが出る（実測）。ブラウザのズームを下げたのと同じ状態を CSS で作る。
   font-size を下げる方法では解決しない。Gradio は余白や高さを px で持っており、
   文字だけ小さくなって全体の高さは変わらないため（こちらも実測で確認）。
   zoom は仕様上は非標準だが Chrome / Edge / Firefox いずれも対応しており、
   ローカルのブラウザで開く前提のツールなので実用上の問題はない。

2. お気に入りボタンの星を描く
   フォントの ★ は字形が環境ごとに変わって見栄えがしない。
   Gradio のボタンはラベルの HTML をエスケープするので SVG を直接は埋め込めない
   （試したところ、タグがそのまま文字として表示された）。
   そこで app.py 側は状態をクラス名で伝えるだけにし、星はここで背景画像として描く。
"""
from __future__ import annotations

# 表示倍率。ブラウザのズーム率と同じ意味で、下げるほど一度に見える量が増える
UI_ZOOM = 90

# 星の形。角を少し丸めて、フォントの ★ より柔らかい印象にしている
_STAR_PATH = (
    "M12 2.6l2.86 5.8 6.4.93-4.63 4.51 1.09 6.37L12 17.11l-5.72 3.1"
    "1.09-6.37L2.74 9.33l6.4-.93z"
)


def _star(fill: str, stroke: str) -> str:
    """data URI に埋め込める形の SVG を作る。fill を変えて塗り／輪郭を作り分ける。"""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
        f"fill='{fill}' stroke='{stroke}' stroke-width='1.7' "
        "stroke-linejoin='round' stroke-linecap='round'>"
        f"<path d='{_STAR_PATH}'/></svg>"
    )
    # URL に置けない文字だけ最小限を置き換える
    return svg.replace("<", "%3C").replace(">", "%3E").replace("#", "%23")


CSS = """
/* 画面全体をブラウザのズーム __ZOOM__% と同じ見え方にする。
   フル HD で縦スクロールを出さずに収めるための調整。 */
gradio-app {
    zoom: __ZOOM__%;
}

/* ---- お気に入りボタンの星 ----
   ラベルは空にしてあり、星はここで背景画像として描く。
   未登録は輪郭だけのグレー、登録済みは塗りつぶしの黄色。 */
.gradio-container button.fav-btn {
    background-image: url("data:image/svg+xml,__STAR_OFF__");
    background-repeat: no-repeat;
    background-position: center;
    background-size: 20px 20px;
    min-width: 46px;
    /* ラベルが空だと、display:flex のボタンは中身が無いぶん高さ 14px まで潰れる
       （文字ありのボタンは 36px）。line-height も min-height も期待どおり効かないので、
       疑似要素で文字と同じ高さの中身を作って押し広げる。これが一番確実だった */
}
.gradio-container button.fav-btn::before {
    content: "";
    display: block;
    /* 文字ありボタンの行の高さ（line-height 24px）と揃える。
       24px + 上下 padding 8px x2 = 36px で他のボタンと同じ高さになる */
    height: 24px;
}
.gradio-container button.fav-btn.on {
    background-image: url("data:image/svg+xml,__STAR_ON__");
}

/* 押せることが分かるよう、触れたときだけ少しだけ大きくする */
.gradio-container button.fav-btn:hover {
    background-size: 22px 22px;
}
@media (prefers-reduced-motion: reduce) {
    .gradio-container button.fav-btn:hover { background-size: 20px 20px; }
}
"""


def build_css() -> str:
    return (
        CSS.replace("__ZOOM__", str(UI_ZOOM))
        .replace("__STAR_OFF__", _star("none", "%23a1a1aa"))
        .replace("__STAR_ON__", _star("%23fbbf24", "%23f59e0b"))
    )
