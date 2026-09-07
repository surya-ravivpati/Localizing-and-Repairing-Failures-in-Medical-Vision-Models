"""Stage the 200 expert-set images + a self-contained browser labeling tool.

Produces results/expert_label/ containing:
  images/<uid>_f.png, <uid>_l.png   (copied frontal/lateral)
  manifest.json                     (blinded: uid, indication, image files)
  label.html                        (offline tool: shows image + 13-finding form,
                                     autosaves to localStorage, exports labels.csv)

Give the whole folder to the radiologist. They open label.html (or run
`python -m http.server` inside the folder if the browser blocks file:// images),
label all studies, click Export, and send back labels.csv for scripts/15.
"""
import json
import os
import shutil

import pandas as pd
from _bootstrap import load_cfg

from src.labeling import CHEXPERT_CLASSES

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]

HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>CXR Expert Labeling</title><style>
body{font-family:system-ui;margin:0;background:#111;color:#eee}
#top{position:sticky;top:0;background:#1c1c1c;padding:10px 16px;display:flex;
gap:16px;align-items:center;border-bottom:1px solid #333}
#imgs{display:flex;gap:8px;justify-content:center;background:#000;padding:8px}
#imgs img{max-height:70vh;max-width:48%;object-fit:contain}
#form{padding:12px 16px;display:grid;grid-template-columns:1fr 1fr;gap:6px 24px}
.row{display:flex;justify-content:space-between;align-items:center;
border-bottom:1px solid #262626;padding:4px 0}
.opts label{margin-left:8px;cursor:pointer;font-size:14px}
button{background:#2d6cdf;color:#fff;border:0;padding:8px 14px;border-radius:6px;
cursor:pointer;font-size:14px}button.sec{background:#444}
#ind{color:#9cf;font-size:14px}.done{color:#4c8}
</style></head><body>
<div id=top>
<button class=sec onclick=go(-1)>&larr; Prev</button>
<span id=pos></span><button class=sec onclick=go(1)>Next &rarr;</button>
<span id=ind></span><span style=flex:1></span>
<span id=prog></span><button onclick=exp()>Export labels.csv</button></div>
<div id=imgs></div><div id=form></div>
<script>
const CLS=%CLASSES%; let M=[], i=0, L={};
fetch('manifest.json').then(r=>r.json()).then(m=>{M=m;
 L=JSON.parse(localStorage.getItem('cxr_labels')||'{}');render()});
function render(){const s=M[i];
 document.getElementById('pos').textContent=(i+1)+' / '+M.length;
 document.getElementById('ind').textContent='Indication: '+(s.indication||'None');
 let h='';for(const f of [s.frontal,s.lateral]) if(f) h+='<img src=images/'+f+'>';
 document.getElementById('imgs').innerHTML=h;
 const cur=L[s.uid]||{};let fh='';
 for(const c of CLS){const v=cur[c]||'absent';
  fh+='<div class=row><span>'+c+'</span><span class=opts>'+
   ['present','uncertain','absent'].map(o=>'<label><input type=radio name="'+c+
    '" value='+o+(v===o?' checked':'')+' onchange=set("'+c+'","'+o+'")>'+o+'</label>').join('')+
   '</span></div>';}
 document.getElementById('form').innerHTML=fh;
 let n=Object.keys(L).length;
 document.getElementById('prog').innerHTML='<span class=done>'+n+'/'+M.length+' saved</span>';}
function set(c,o){const u=M[i].uid;L[u]=L[u]||{};L[u][c]=o;
 localStorage.setItem('cxr_labels',JSON.stringify(L));}
function go(d){if(!L[M[i].uid])L[M[i].uid]=L[M[i].uid]||seed();i=Math.max(0,
 Math.min(M.length-1,i+d));localStorage.setItem('cxr_labels',JSON.stringify(L));render();}
function seed(){const o={};for(const c of CLS)o[c]='absent';return o;}
function exp(){let rows=['uid,'+CLS.map(c=>'label_'+c).join(',')];
 for(const s of M){const l=L[s.uid]||seed();
  rows.push(s.uid+','+CLS.map(c=>l[c]||'absent').join(','));}
 const b=new Blob([rows.join('\\n')],{type:'text/csv'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);
 a.download='labels.csv';a.click();}
</script></body></html>"""


def main():
    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    man = pd.read_csv(os.path.join(out, "expert_manifest.csv"))
    dst = os.path.join(out, "expert_label")
    img_dir = os.path.join(dst, "images")
    os.makedirs(img_dir, exist_ok=True)

    entries, copied = [], 0
    for _, r in man.iterrows():
        uid = r["uid"]
        e = {"uid": int(uid), "indication": str(r.get("indication", "") or "")}
        for side, col in [("frontal", "frontal_path"), ("lateral", "lateral_path")]:
            p = r.get(col, "")
            if isinstance(p, str) and p and os.path.exists(p):
                fn = f"{uid}_{side[0]}.png"
                shutil.copy(p, os.path.join(img_dir, fn))
                e[side] = fn
                copied += 1
            else:
                e[side] = ""
        entries.append(e)

    json.dump(entries, open(os.path.join(dst, "manifest.json"), "w"))
    html = HTML.replace("%CLASSES%", json.dumps(ALL))
    open(os.path.join(dst, "label.html"), "w").write(html)
    print(f"staged {len(entries)} studies, {copied} images -> {dst}")
    print(f"  give the folder to the radiologist; they open label.html "
          f"(or `python -m http.server` inside it), label, and Export labels.csv")


if __name__ == "__main__":
    main()
