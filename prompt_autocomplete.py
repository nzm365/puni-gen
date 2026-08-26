"""プロンプト欄に Danbooru タグのインライン補完を付ける（head に注入する JS/CSS）。

data/danbooru_tags.csv（build_tags.py で生成）を読み、対象の textarea に
入力補完のポップアップを出す。読み込み済み embedding のトリガーワードも
候補に混ぜる。ネットワークも外部ライブラリも実行時には使わない。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

TAGS_CSV = Path(__file__).parent / "data" / "danbooru_tags.csv"

# 補完対象の textarea（app.py 側で elem_id を合わせる）
TARGET_IDS = ["prompt_box", "neg_box"]

# category → (色, ラベル)。Danbooru の区分に合わせる。
# イラストレーター名 (category 1) は辞書生成の段階で除外しているのでここにも持たない
CAT_META = {
    0: ("#8cc4ff", "一般"),
    3: ("#d6a5f5", "版権"),
    4: ("#8ee6a0", "キャラ"),
    6: ("#f5d264", "埋込"),   # このツール独自: embedding のトリガーワード
}


def _js_json(obj) -> str:
    """<script> の中に直接埋め込める JSON を作る。

    json.dumps は "<" や "/" をエスケープしないので、値に "</script>" が含まれると
    そこでスクリプトが閉じてしまい、任意の HTML を差し込まれる。
    embedding のファイル名もタグ辞書も外部から来る文字列なので、
    閉じタグを構成できない形に落としてから埋め込む。
    """
    return (
        json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("&", r"\u0026")
    )


def _load_tags() -> list[list]:
    if not TAGS_CSV.exists():
        return []
    rows: list[list] = []
    with TAGS_CSV.open(encoding="utf-8", newline="") as f:
        for name, cat, count in csv.reader(f):
            rows.append([name, int(cat), int(count)])
    return rows


def build_head(embeddings: list[str] | None = None) -> str:
    """Blocks(head=...) に渡す HTML を返す。タグ辞書が無ければ空文字。"""
    tags = _load_tags()
    if not tags:
        return ""
    # embedding のトリガーワードを最優先で先頭に。count は最大扱いで常に上位に出す
    data = [[e, 6, 1 << 62] for e in (embeddings or [])] + tags
    payload = _js_json(data)
    cfg = _js_json({
        "ids": TARGET_IDS,
        "cats": {str(k): v[0] for k, v in CAT_META.items()},
        "max": 12,
    })
    return _TEMPLATE.replace("/*__DATA__*/", payload).replace("/*__CFG__*/", cfg)


_TEMPLATE = r"""
<style>
#ac-box{position:absolute;z-index:10000;display:none;max-height:280px;overflow-y:auto;
  min-width:260px;background:#1f2024;border:1px solid #4a4b52;border-radius:8px;
  box-shadow:0 8px 24px rgba(0,0,0,.5);font-size:13px;padding:4px}
#ac-box .ac-item{display:flex;justify-content:space-between;gap:12px;align-items:center;
  padding:5px 9px;border-radius:5px;cursor:pointer;white-space:nowrap}
#ac-box .ac-item.sel{background:#37507a}
#ac-box .ac-name{overflow:hidden;text-overflow:ellipsis}
#ac-box .ac-count{color:#9aa0aa;font-size:11px;flex:none}
</style>
<script>
(function(){
  var DATA=/*__DATA__*/;
  var CFG=/*__CFG__*/;
  var box=null, items=[], sel=-1, active=null;

  function fmt(n){
    if(n>=1e6) return (n/1e6).toFixed(1).replace(/\.0$/,'')+'M';
    if(n>=1e3) return (n/1e3).toFixed(0)+'k';
    return ''+n;
  }
  function isTarget(el){
    if(!el||el.tagName!=='TEXTAREA') return false;
    for(var i=0;i<CFG.ids.length;i++){ if(el.closest('#'+CFG.ids[i])) return true; }
    return false;
  }
  function ensureBox(){
    if(box) return box;
    box=document.createElement('div'); box.id='ac-box';
    document.body.appendChild(box);
    box.addEventListener('mousedown',function(e){ e.preventDefault(); }); // blur を防ぐ
    return box;
  }
  // カーソル直前の「今書いているタグ」（直近のカンマ/改行以降）を取り出す
  function currentFragment(ta){
    var upto=ta.value.slice(0,ta.selectionStart);
    var m=upto.split(/[,\n]/);
    return m[m.length-1];
  }
  function search(q){
    var norm=q.trim().toLowerCase().replace(/ /g,'_');
    if(!norm) return [];
    var out=[];
    for(var i=0;i<DATA.length && out.length<CFG.max;i++){
      if(DATA[i][0].toLowerCase().indexOf(norm)===0) out.push(DATA[i]);
    }
    return out;
  }
  function render(list,ta){
    var b=ensureBox();
    items=list; sel=list.length?0:-1;
    if(!list.length){ b.style.display='none'; return; }
    b.innerHTML='';
    list.forEach(function(t,idx){
      var color=CFG.cats[t[1]]||'#ddd';
      var row=document.createElement('div');
      row.className='ac-item'+(idx===0?' sel':'');
      row.innerHTML='<span class="ac-name" style="color:'+color+'">'+
        t[0].replace(/</g,'&lt;')+'</span><span class="ac-count">'+fmt(t[2])+'</span>';
      row.addEventListener('click',function(){ insert(ta,items[idx]); });
      b.appendChild(row);
    });
    var r=ta.getBoundingClientRect();
    b.style.left=(window.scrollX+r.left)+'px';
    b.style.top=(window.scrollY+r.bottom+4)+'px';
    b.style.display='block';
  }
  function hide(){ if(box) box.style.display='none'; items=[]; sel=-1; }
  function move(d){
    if(!items.length) return;
    var rows=box.querySelectorAll('.ac-item');
    rows[sel] && rows[sel].classList.remove('sel');
    sel=(sel+d+items.length)%items.length;
    rows[sel].classList.add('sel');
    rows[sel].scrollIntoView({block:'nearest'});
  }
  // タグ名 → プロンプトに書く形。アンダースコアは空白へ、括弧はエスケープ
  // （括弧は強調構文なので、素の () だと重み付けと誤解釈される）
  function toPrompt(name){
    return name.replace(/_/g,' ').replace(/[()]/g,function(c){ return '\\'+c; });
  }
  function insert(ta,tag){
    var start=ta.selectionStart;
    var before=ta.value.slice(0,start), after=ta.value.slice(start);
    var sepIdx=Math.max(before.lastIndexOf(','),before.lastIndexOf('\n'));
    var head=before.slice(0,sepIdx+1);
    var lead=before.slice(sepIdx+1).match(/^\s*/)[0]; // 直後の空白は保つ
    var text=toPrompt(tag[0])+', ';
    ta.value=head+lead+text+after.replace(/^\s*/,'');
    var pos=(head+lead+text).length;
    ta.selectionStart=ta.selectionEnd=pos;
    ta.focus();
    ta.dispatchEvent(new Event('input',{bubbles:true})); // Gradio に変更を伝える
    hide();
  }
  document.addEventListener('input',function(e){
    if(!isTarget(e.target)){ return; }
    active=e.target;
    var frag=currentFragment(e.target);
    if(frag.trim().length<1){ hide(); return; }
    render(search(frag),e.target);
  });
  document.addEventListener('keydown',function(e){
    if(!box || box.style.display==='none' || e.target!==active) return;
    if(e.key==='ArrowDown'){ e.preventDefault(); move(1); }
    else if(e.key==='ArrowUp'){ e.preventDefault(); move(-1); }
    else if(e.key==='Enter'||e.key==='Tab'){
      if(sel>=0){ e.preventDefault(); e.stopPropagation(); insert(e.target,items[sel]); }
    }
    else if(e.key==='Escape'){ e.preventDefault(); hide(); }
  },true);
  document.addEventListener('click',function(e){
    if(box && e.target!==active && !box.contains(e.target)) hide();
  });
})();
</script>
"""
