"""画面まわりの CSS（Gradio の launch(css=...) に渡す）。

やっていることは 4 つ。

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

3. 進捗バーを描く
   待ちには 2 種類ある。step 数のように残りが数えられるものは、実際の割合まで
   伸びるバーで出す。モデルの読み込みのように数えられないものは、割合を偽らずに
   丸い塊を左右に往復させて「止まっていない」ことだけを示す。
   往復の折り返しで横に伸び縮みさせているのは、PuniGen の名前のとおり
   柔らかい印象にするため（縞模様も試したが、角が立って固い見た目になった）。
   app.py が中身を空にすると要素ごと消えるので、待ちが無いときは場所を取らない。

4. モデル選択と再読み込みボタンを 1 つの枠に見せる
   Gradio はコンポーネントごとに枠を描くので、素直に並べると枠が 2 つできる。
   行 (#model_row) に枠を移し、中の 2 つからは枠と背景を外す。
   ボタンは幅を詰めて、選択欄の右端に収まるようにする。
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

/* ---- モデル選択 + 再読み込みボタン ----
   枠を行に移し、中身は枠なしにして、1 つの入力欄の中にボタンがあるように見せる。 */
#model_row {
    border: 1px solid var(--block-border-color, var(--border-color-primary, #d4d4d8));
    border-radius: var(--block-radius, 8px);
    background: var(--block-background-fill, transparent);
    /* 下の余白は行側に持たせる。align-items: flex-end なので選択欄とボタンは
       この余白の上で下端がそろい、囲いの内側にも同じだけ空きができる */
    padding: 0 var(--size-1, 4px) var(--block-padding, 8px) 0;
    /* 囲いと次の入力欄の間にも間隔を空ける */
    margin-bottom: var(--size-2, 8px);
    gap: 0;
    flex-wrap: nowrap;
    /* 中央ではなく下端に揃える。選択欄の上には「モデル」のラベルがあるので、
       中央だとボタンだけがラベルのぶん浮いて見える */
    align-items: flex-end;
}
/* 中の 2 つは自前の枠・背景・影を持たない */
#model_row .block,
#model_row .form,
#model_row .container {
    border: none;
    background: transparent;
    box-shadow: none;
}
/* 選択欄の下端を行の下端に一致させる。これでボタンと高さが揃う */
#model_row .block {
    padding-bottom: 0;
}
/* 再読み込みボタン。幅は文字 1 つぶんまで詰めるが、枠と地色は残す。
   枠なしにすると、ただの記号に見えて押せることが伝わらなかった */
#model_row button {
    min-width: 0;
    width: 2.3rem;
    flex: 0 0 2.3rem;
    /* 高さを決め打ちせず、入力欄と同じ「縦 padding + 1 行」で組み立てる。
       同じ計算で高さが出るので、テーマが変わってもずれない */
    padding: var(--input-padding, 8px 10px);
    padding-left: 0;
    padding-right: 0;
    border: 1px solid var(--button-secondary-border-color, var(--border-color-primary, #d4d4d8));
    border-radius: var(--radius-sm, 6px);
    background: var(--button-secondary-background-fill, #f4f4f5);
    box-shadow: var(--shadow-drop, 0 1px 2px rgb(0 0 0 / 8%));
    color: var(--body-text-color, #27272a);
    font-size: 1.05rem;
    line-height: var(--line-sm, 1.25);
    cursor: pointer;
}
#model_row button:hover {
    background: var(--button-secondary-background-fill-hover, #e4e4e7);
    border-color: var(--body-text-color-subdued, #a1a1aa);
}
/* 押した瞬間に沈める。反応があったことを触感として返す */
#model_row button:active {
    transform: translateY(1px);
    box-shadow: none;
}

/* ---- 進捗バー ----
   app.py の progress（左カラム・モデルの状態表示の直下）専用。
   「いま何をしているか」はここだけに出る。
   空文字を入れると中身ごと消えるので、待ちが無いときは行が畳まれる。 */
#progress_line {
    min-height: 0;
    /* status と左端をそろえる */
    padding: 0;
}

/* Gradio はジェネレータの実行中、出力コンポーネントの枠を 2 秒周期で点滅させる
   (statustracker の .generating)。この行はバー自体が動いているので枠は情報を足さず、
   二重に主張してうるさいだけなので止める */
#progress_line .generating {
    border: none !important;
    animation: none !important;
}
#progress_line .pg {
    margin: 0 0 6px;
}
#progress_line .pg-text {
    font-size: 0.9em;
    color: var(--body-text-color-subdued, #71717a);
    margin-bottom: 4px;
}
#progress_line .pg-track {
    height: 8px;
    border-radius: 999px;
    background: var(--background-fill-secondary, #ececf1);
    overflow: hidden;
}
#progress_line .pg-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #fdba74, var(--color-accent, #f97316));
    transition: width 0.25s ease;
}

/* 残りが数えられない待ち。丸い塊が左右を往復する。
   translateX と scaleX だけで動かすので、毎フレームのレイアウト計算が要らない。
   幅 32% の塊を自分の幅の 212% ぶん動かすと 32 + 32*2.12 = 99.8%、
   ちょうど端から端まで届く。折り返しは alternate に任せる。
   中間で scaleX を 1.35 に膨らませ、伸びて縮む柔らかい動きにしている */
#progress_line .pg-ind .pg-fill {
    width: 32%;
    transform-origin: center;
    animation: pg-puni 1.3s cubic-bezier(0.45, 0, 0.55, 1) infinite alternate;
}
@keyframes pg-puni {
    from { transform: translateX(0) scaleX(1); }
    50%  { transform: translateX(106%) scaleX(1.35); }
    to   { transform: translateX(212%) scaleX(1); }
}

/* 終わったあと。満杯のバーを緑にして残す。
   消してしまうと、待たされていた人に結果が残らない */
#progress_line .pg-done .pg-fill {
    background: linear-gradient(90deg, #86efac, #22c55e);
}
#progress_line .pg-done .pg-text {
    color: var(--body-text-color, #27272a);
}

/* 一言だけのとき（中止しました、など）はバーを出さない */
#progress_line .pg-note .pg-text {
    margin-bottom: 0;
}

@media (prefers-reduced-motion: reduce) {
    /* 動きを止める設定では往復させず、控えめな塊を中央に置くだけにする */
    #progress_line .pg-ind .pg-fill {
        animation: none;
        transform: translateX(106%);
        opacity: 0.55;
    }
    #progress_line .pg-fill { transition: none; }
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
