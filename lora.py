"""LoRA の一覧と、プロンプトに書かれた `<lora:name:0.8>` の解釈。

ここには UI もパイプラインも出てこない。ファイルを数えることと文字列を解釈すること
だけを持たせてあるので、GPU もモデルも無い状態で動かして確かめられる。
実際にパイプラインへ載せるのは engine.py、画面に並べるのは app.py の担当。
"""
from __future__ import annotations

import re
from pathlib import Path

LORA_DIR = Path(__file__).parent / "models" / "loras"

# 同時に使える数の上限。VRAM の都合で決めた値ではない。
# 4 個以上を重ねると効果が互いに混ざり、どのスライダーがどの変化に効いているのかを
# 追えなくなる。数を絞ったほうが結果を制御できる、という判断でこの値にしている。
# VRAM が理由ではないので、大きな GPU を積んでも上限は変わらない。
MAX = 3

# 強度の範囲。0 は「読み込むが効かせない」、1.5 は効かせすぎて絵が壊れ始める手前。
WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.5
WEIGHT_DEFAULT = 0.8

# Civitai などからコピーしてくるプロンプトに混ざる記法。
#   <lora:name:0.8>  重みつき
#   <lora:name>      重み省略（既定値を使う）
# name はファイル名がそのまま入るので、区切りに使う : < > 以外は通す。
_TAG = re.compile(r"<lora:([^:<>]+?)(?::([+-]?\d*\.?\d+))?>", re.IGNORECASE)

# タグを抜いた跡に残る区切りの後始末に使う
_DOUBLE_COMMA = re.compile(r",(?:\s*,)+")
_SPACES = re.compile(r"[^\S\n]{2,}")


def list_loras() -> list[str]:
    """models/loras/ にある LoRA を名前で返す。

    拡張子を落とした名前で扱う。`<lora:name:0.8>` の name がこの形なので、
    プロンプトからのコピペとそのまま突き合わせられる。
    （チェックポイントは拡張子つきで扱っているが、あちらはタグ記法を持たない）
    """
    if not LORA_DIR.exists():
        return []
    return sorted(p.stem for p in LORA_DIR.glob("*.safetensors"))


def path_of(name: str) -> Path | None:
    """名前から実ファイルを引く。無ければ None。"""
    p = LORA_DIR / f"{name}.safetensors"
    return p if p.exists() else None


def total_bytes(names) -> int:
    """選択中の LoRA の合計サイズ。VRAM の事前判定に足すために使う。"""
    total = 0
    for name in names:
        p = path_of(name)
        if p is not None:
            total += p.stat().st_size
    return total


def clamp(weight: float) -> float:
    """強度をスライダーの範囲に収める。範囲外の値がコピペで来ることがある。"""
    return max(WEIGHT_MIN, min(WEIGHT_MAX, float(weight)))


def extract(prompt: str) -> tuple[str, list[tuple[str, float]]]:
    """プロンプトから `<lora:...>` を抜き出す。(残りのプロンプト, [(名前, 強度)])。

    抜くのはこの記法だけで、トリガーワードなど他の語には触らない。
    抜いた跡に「, ,」や連続した空白が残るので、そこだけ詰める。
    """
    if not prompt:
        return prompt or "", []

    found: list[tuple[str, float]] = []

    def take(m: re.Match) -> str:
        raw = m.group(2)
        weight = clamp(raw) if raw is not None else WEIGHT_DEFAULT
        found.append((m.group(1).strip(), weight))
        return ""

    rest = _TAG.sub(take, prompt)
    if found:  # 何も抜いていないなら、区切りの詰め直しもしない
        rest = _DOUBLE_COMMA.sub(",", rest)
        rest = _SPACES.sub(" ", rest)
        rest = rest.strip().strip(",").strip()
    return rest, found


def resolve(items) -> tuple[list[tuple[str, float]], list[str]]:
    """(名前, 強度) の並びを、手元にあるものと無いものに分ける。

    無いものを黙って捨てないのは、コピペ元では効いていた LoRA が
    抜け落ちたまま生成され、絵が変わった理由が分からなくなるため。
    呼び出し側は未解決の名前を画面に出す。

    同じ LoRA が二度出てきたら後ろを捨てる。同じアダプタ名は一度しか登録できない。
    """
    found: list[tuple[str, float]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for name, weight in items:
        if name in seen:
            continue
        seen.add(name)
        if path_of(name) is None:
            missing.append(name)
        else:
            found.append((name, clamp(weight)))
    return found, missing
