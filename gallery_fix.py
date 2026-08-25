"""ギャラリーの疑似フルスクリーンを、閉じる操作で確実に解除する（head に注入する JS）。

Gradio 6.25 の Gallery は、フルスクリーンボタンでブラウザの Fullscreen API を呼ばず、
ラッパー要素に `fullscreen` クラスを付けて画面いっぱいに広げる実装になっている。
このとき「閉じる (Close)」を押すと拡大表示だけが閉じ、`fullscreen` クラスは残る。
さらに格子表示へ戻るとフルスクリーン解除ボタン自体が消えるため、
画面いっぱいのまま戻れなくなる（実測で確認: 1216x720 -> 1385x950 のまま固定）。

そこで「閉じる」が処理される前に、Gradio 自身の解除ボタンを押しておく。
クラスを直接消すのではなく公式のボタンを経由するので、内部状態とずれない。
"""
from __future__ import annotations

# 解除ボタンの aria-label（Gradio が付与するもの）
EXIT_LABEL = "Exit fullscreen mode"
CLOSE_LABEL = "Close"

_TEMPLATE = """
<script>
(function () {
  var EXIT = 'button[aria-label="__EXIT__"]';
  var CLOSE = 'button[aria-label="__CLOSE__"]';

  // alsoClose: 拡大表示も自分で閉じる（Escape 用）。
  // 解除ボタンを click すると再描画が走り、Gradio 側の Escape 処理が流れてしまうため、
  // その場合は次のタックで閉じるボタンを押して拡大表示も畳む。
  function leaveFullscreen(alsoClose) {
    var exit = document.querySelector(EXIT);
    var wasFullscreen = !!exit;
    if (exit) exit.click();               // Gradio 自身の解除処理を通す
    if (document.fullscreenElement) {     // 本物の全画面なら併せて解除
      document.exitFullscreen();
    }
    if (alsoClose && wasFullscreen) {
      setTimeout(function () {
        var c = document.querySelector(CLOSE);
        if (c) c.click();
      }, 0);
    }
  }

  // capture 段階で拾い、Gradio の閉じる処理より先に解除する。
  // 閉じたあとでは解除ボタンが DOM から消えるため、先回りする必要がある。
  document.addEventListener('click', function (e) {
    var t = e.target;
    if (!t || !t.closest) return;
    if (t.closest(CLOSE)) leaveFullscreen(false);  // 閉じるは Gradio 側が行う
  }, true);

  // Escape でも拡大表示は閉じるので、同じ後始末をする
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') leaveFullscreen(true);
  }, true);
})();
</script>
"""


def build_head() -> str:
    return _TEMPLATE.replace("__EXIT__", EXIT_LABEL).replace("__CLOSE__", CLOSE_LABEL)
