"""画面まわりの CSS（Gradio の launch(css=...) に渡す）。

やっていることは 5 つ。

1. 表示倍率を下げる
   Gradio の既定は余白も文字も大きめで、フル HD だとページ全体が 1070px になり
   縦スクロールが出る（実測）。ブラウザのズームを下げたのと同じ状態を CSS で作る。
   font-size を下げる方法では解決しない。Gradio は余白や高さを px で持っており、
   文字だけ小さくなって全体の高さは変わらないため（こちらも実測で確認）。
   zoom は仕様上は非標準だが Chrome / Edge / Firefox いずれも対応しており、
   ローカルのブラウザで開く前提のツールなので実用上の問題はない。

2. お気に入りの星と削除のゴミ箱を描く
   フォントの ★ や 🗑 は字形が環境ごとに変わって見栄えがしない。
   Gradio のボタンはラベルの HTML をエスケープするので SVG を直接は埋め込めない
   （試したところ、タグがそのまま文字として表示された）。
   そこで app.py 側は状態をクラス名で伝えるだけにし、絵はここで背景画像として描く。

3. 進捗バーを描く
   待ちには 2 種類ある。step 数のように残りが数えられるものは、実際の割合まで
   伸びるバーで出す。モデルの読み込みのように数えられないものは、割合を偽らずに
   丸い塊を左右に往復させて「止まっていない」ことだけを示す。
   往復の折り返しで横に伸び縮みさせているのは、PuniGen の名前のとおり
   柔らかい印象にするため（縞模様も試したが、角が立って固い見た目になった）。
   app.py が中身を空にすると要素ごと消えるので、待ちが無いときは場所を取らない。

4. モデル選択と再読み込みボタンを 1 つの入力グループに見せる
   Dropdown が自分で描く枠は「ドロップダウンだ」と分かるための枠なので消さない。
   行にもう 1 つ枠を足すと二重になるので、行は枠を持たず、
   ボタンを選択欄の右へ密着させて、まとまった 1 つの部品に見せる。

5. 効かない「共有」ボタンを隠す
   画像を開くと右上に出るが、これは Gradio の Hugging Face Spaces へ共有する
   機能で、Spaces 上でしか成立しない。ローカルで押すと Share failed. が出るだけ。
   Gallery 側に出し分けの引数が無い（Gradio 6 で無くなった）ので CSS で隠す。
"""
from __future__ import annotations

# 表示倍率。ブラウザのズーム率と同じ意味で、下げるほど一度に見える量が増える
UI_ZOOM = 90

# 再読み込みボタンの高さ。選択欄の高さに合わせる値。
# 選択欄の高さは --checkbox-label-padding（テーマ側で決まり、CSS ファイルからは
# 読めない）に依存するため計算では出せない。ここだけ実測で合わせる
REFRESH_BTN_HEIGHT = "42px"

# 星の形。角を少し丸めて、フォントの ★ より柔らかい印象にしている
_STAR_PATH = (
    "M12 2.6l2.86 5.8 6.4.93-4.63 4.51 1.09 6.37L12 17.11l-5.72 3.1"
    "1.09-6.37L2.74 9.33l6.4-.93z"
)


# ゴミ箱。ふた・本体・中の 2 本線を 1 本の path にまとめてある
_TRASH_PATH = (
    "M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4"
    "a2 2 0 0 1 2 2v2M10 11v6M14 11v6"
)


def _svg(path: str, fill: str, stroke: str, width: str = "1.7") -> str:
    """data URI に埋め込める形の SVG を作る。fill を変えて塗り／輪郭を作り分ける。"""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
        f"fill='{fill}' stroke='{stroke}' stroke-width='{width}' "
        "stroke-linejoin='round' stroke-linecap='round'>"
        f"<path d='{path}'/></svg>"
    )
    # URL に置けない文字だけ最小限を置き換える
    return svg.replace("<", "%3C").replace(">", "%3E").replace("#", "%23")


def _star(fill: str, stroke: str) -> str:
    return _svg(_STAR_PATH, fill, stroke)


def _trash(stroke: str) -> str:
    # 塗らない。星の未登録と同じ「輪郭だけ」の見え方に揃える
    return _svg(_TRASH_PATH, "none", stroke, "1.8")


CSS = """
/* 画面全体をブラウザのズーム __ZOOM__% と同じ見え方にする。
   フル HD で縦スクロールを出さずに収めるための調整。 */
gradio-app {
    zoom: __ZOOM__%;
}

/* ---- 効かない「共有」ボタンを隠す ----
   画像を開いたときの右上に出る。押しても Share failed. のトーストが出るだけで、
   ローカル実行では成立しない機能なので出さない（実際に押して確認した）。
   英語表記も併記しているのは、Gradio の翻訳が揃っておらず、同じ並びに
   「ダウンロード」と「Fullscreen」が混在するなど、環境で表記が変わるため。
   ダウンロード / Fullscreen / 閉じるはそのまま残す。 */
.gradio-container button[aria-label="共有"],
.gradio-container button[aria-label="Share"] {
    display: none;
}

/* ---- モデル選択 + 再読み込みボタン ----
   「プレフィックス」などの入力欄と同じ見た目にする。Gradio のあの枠は border ではなく
   .block padded の地色 + 内側の余白なので、同じ変数を行に当てて同じ箱を作る。
   中の Dropdown からは箱としての役割を外し、選択欄の枠 (.container .wrap) だけ残す。
   こうすると、ラベル・選択欄・ボタンが 1 つの箱に収まり、他の入力欄と揃う。 */
#model_row {
    background: var(--block-background-fill);
    border: var(--block-border-width, 0px) solid var(--block-border-color, transparent);
    border-radius: var(--block-radius, 8px);
    padding: var(--block-padding, 10px 12px);
    gap: var(--spacing-sm, 4px);
    flex-wrap: nowrap;
    /* ラベル「モデル」のぶん選択欄は下寄りにあるので、下端で揃える */
    align-items: flex-end;
    margin-bottom: var(--size-2, 8px);
}
/* 中の入れ物は箱の役割を持たない。地色も余白も外側の行が持つ */
#model_row .block,
#model_row .form {
    background: transparent;
    border: none;
    box-shadow: none;
    padding: 0;
}

/* 再読み込みボタン。幅は文字 1 つぶんまで詰め、高さは選択欄に合わせる。
   高さは --checkbox-label-padding 依存でテーマ側でしか決まらず計算で出せないため、
   ブラウザで実測して REFRESH_BTN_HEIGHT に置いている */
#model_row button {
    min-width: 0;
    width: 2.3rem;
    flex: 0 0 2.3rem;
    height: __BTN_H__;
    padding: 0;
    border: var(--input-border-width, 1px) solid var(--border-color-primary, #3f3f46);
    border-radius: var(--input-radius, 4px);
    background: var(--button-secondary-background-fill, #3f3f46);
    box-shadow: var(--input-shadow, none);
    color: var(--body-text-color, #e4e4e7);
    font-size: 1.05rem;
    line-height: 1;
    cursor: pointer;
}
#model_row button:hover {
    background: var(--button-secondary-background-fill-hover, #52525b);
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
   (statustracker の .generating: 2px の枠 + 明滅)。
   生成の進み具合はこの進捗バーが受け持っているので、同じことを枠の明滅でも
   主張されると二重になる。特に「生成情報」は生成中は空のまま光り、
   結果ギャラリーは大きな橙の矩形になって、進捗バーと離れた位置で目立ってしまう。
   バーが受け持つ 3 つだけ明滅を止める。検索結果のように、バーが無くて
   この明滅が唯一の反応になる箇所はそのまま残す */
#progress_line .generating,
#result_gallery .generating,
#gen_info .generating {
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
    /* 登録済みのときの variant="primary" も、触れたときの :hover も、Gradio は
       background の一括指定で当ててくる。指定されると絵そのものが none に
       戻って消え、repeat / position / size も初期値に戻って絵が一面に
       敷き詰められる（どちらも実際にそうなっていた）。ここは全部譲らない */
    background-image: url("data:image/svg+xml,__STAR_OFF__") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 20px 20px !important;
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
    background-image: url("data:image/svg+xml,__STAR_ON__") !important;
}


/* ---- 削除ボタンのゴミ箱 ----
   星と同じ作り。ラベルの「削除」は下の指定で見えなくしてあるが、消してはいない。
   読み上げソフトに押せるものの名前が残り、CSS が届かない環境では文字が出る。 */
.gradio-container button.del-btn {
    background-image: url("data:image/svg+xml,__TRASH__") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 19px 19px !important;
    min-width: 46px;
    /* 文字は消さず、見えなくするだけ。幅も食わないよう 0 にする */
    font-size: 0;
    color: transparent;
}
.gradio-container button.del-btn::before {
    content: "";
    display: block;
    /* 文字ありのボタンと同じ 36px の高さにする（星と同じ理由） */
    height: 24px;
}

/* ---- 絵のボタンに触れたとき ----
   絵ではなくボタンの地色を変える。絵は 20px 前後と小さく、そこだけ色を変えても
   変化に気づきにくいため。面で変えた方が、どのボタンの上にいるか一目で分かる。

   色は地色を置き換えるのではなく、ボタンの地色の上に半透明の面を 1 枚重ねる。
   background-color を差し替えると、ボタンが持っていた不透明な地色ごと消えて
   下の画面が透け、明るくなるはずのホバーが逆に暗く沈んで見えた。
   絵はその面より上に置くので、色を重ねても絵は隠れない。

   お気に入り済みのときは variant="primary" が付き、Gradio の .primary は
   background を一括指定で当ててくる。ここも !important で譲らない
   （星の背景画像を守っているのと同じ事情）。

   赤も黄も常時ではなく触れたときだけにしている。ゴミ箱の実体は
   outputs/.trash/ への移動で、フォルダから出せば戻せる。常に赤い面を置くと、
   取り返しがつかない操作に見えてしまう。 */
.gradio-container button.fav-btn:hover {
    background-image:
        url("data:image/svg+xml,__STAR_OFF__"),
        linear-gradient(__FAV_TINT__, __FAV_TINT__) !important;
    background-repeat: no-repeat, no-repeat !important;
    background-position: center, center !important;
    background-size: 20px 20px, 100% 100% !important;
}
/* 登録済みのときは、ボタンがもう黄色く塗られている。そこへ同じ黄色を重ねても
   色が濁るだけでほとんど変わらなかった（実測で差 45、しかも彩度が落ちる方向）。
   塗りつぶしのボタンは暗くするのが分かりやすいので、こちらだけ黒を重ねる。 */
.gradio-container button.fav-btn.on:hover {
    background-image:
        url("data:image/svg+xml,__STAR_ON__"),
        linear-gradient(__FAV_ON_TINT__, __FAV_ON_TINT__) !important;
}
.gradio-container button.del-btn:hover {
    background-image:
        url("data:image/svg+xml,__TRASH__"),
        linear-gradient(__DEL_TINT__, __DEL_TINT__) !important;
    background-repeat: no-repeat, no-repeat !important;
    background-position: center, center !important;
    background-size: 19px 19px, 100% 100% !important;
}
"""


def build_css() -> str:
    return (
        CSS.replace("__ZOOM__", str(UI_ZOOM))
        .replace("__BTN_H__", REFRESH_BTN_HEIGHT)
        .replace("__STAR_OFF__", _star("none", "%23a1a1aa"))
        .replace("__STAR_ON__", _star("%23fbbf24", "%23f59e0b"))
        .replace("__FAV_TINT__", "rgba(251, 191, 36, 0.30)")
        .replace("__FAV_ON_TINT__", "rgba(0, 0, 0, 0.22)")
        .replace("__DEL_TINT__", "rgba(239, 68, 68, 0.30)")
        .replace("__TRASH__", _trash("%23a1a1aa"))
    )
