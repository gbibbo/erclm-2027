"""Blind listening page for the provenance negatives (W4 of the review of draft v4).

Copies the 1000 ms variant of the 114 non-speech windows of the provenance bank under neutral
names, in a fixed random order, and writes a local HTML page to label them. The page shows no
tagger score and no clip name. Labels are kept in the browser and downloaded as a CSV, which
tools/blind_listening.py --read joins back to the clips. Nothing is uploaded and nothing is
committed: the page and the audio live in _work/closing/listening.

    PYTHONPATH=src python tools/blind_listening.py            # build the page
    PYTHONPATH=src python tools/blind_listening.py --read F   # join a downloaded CSV to the clips
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closing_analyses import WORK, md  # noqa: E402

BANK = WORK / "prov_audio" / "prov_bank_v3"
OUT = WORK / "listening"
LABELS = [("voz", "Clear speech"), ("voz_posible", "Possible or background speech"),
          ("vocal_no_verbal", "Singing, laughter or another vocal sound"), ("sin_voz", "No speech")]

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind listening to negatives</title>
<style>
:root { --bg:#f7f7f5; --card:#ffffff; --ink:#1d1d1b; --muted:#6b6b66; --line:#deded8; --accent:#2a78d6; --done:#1baf7a; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --card:#21211f; --ink:#ecece8; --muted:#a3a39c; --line:#3a3a36; --accent:#5b9ce6; --done:#3cc993; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:640px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:20px; margin:0 0 4px; }
p.lead { color:var(--muted); margin:0 0 20px; font-size:14px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; }
.top { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px; }
.count { font-size:28px; font-weight:600; font-variant-numeric:tabular-nums; }
.bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; margin-bottom:16px; }
.bar > div { height:100%; background:var(--done); width:0; transition:width .2s; }
button { font:inherit; color:var(--ink); background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 14px; cursor:pointer; }
button:hover { border-color:var(--accent); }
button.sel { border-color:var(--accent); box-shadow:inset 0 0 0 2px var(--accent); }
.play { width:100%; font-size:18px; padding:14px; margin-bottom:16px; }
.labels { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px; }
.labels kbd { display:inline-block; min-width:20px; color:var(--muted); font:13px ui-monospace, monospace; margin-right:6px; }
input[type=text] { width:100%; font:inherit; color:var(--ink); background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:8px 10px; margin-bottom:12px; }
.nav { display:flex; gap:8px; justify-content:space-between; flex-wrap:wrap; }
.muted { color:var(--muted); font-size:13px; }
@media (max-width:420px) { .labels { grid-template-columns:1fr; } }
</style>
</head>
<body>
<main>
<h1>Blind listening to negatives</h1>
<p class="lead">Each clip lasts one second and comes from a VoxConverse region annotated as non-speech. Choose the option that describes it best.
Keys: <b>space</b> replays, <b>1 to 4</b> label and move on, <b>left arrow</b> goes back. Progress is saved in this browser.</p>
<div class="card">
  <div class="top"><span class="count" id="count"></span><span class="muted" id="state"></span></div>
  <div class="bar"><div id="bar"></div></div>
  <audio id="audio" preload="auto"></audio>
  <button class="play" id="play">&#9654; Listen</button>
  <div class="labels" id="labels"></div>
  <input type="text" id="note" placeholder="Optional note (for example: distant speech at the end)">
  <div class="nav">
    <button id="prev">&#8592; Previous</button>
    <button id="next">Next unlabelled &#8594;</button>
    <button id="dl">Download CSV</button>
  </div>
  <p class="muted" id="summary"></p>
</div>
</main>
<script>
const CLIPS = __CLIPS__;
const LABELS = __LABELS__;
const KEY = "listening-negatives-v1";
let st = { i: 0, ans: {} };
try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); if (s && s.ans) st = s; } catch (e) {}
const $ = id => document.getElementById(id);
const audio = $("audio");
function save() { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} }
function done() { return CLIPS.filter(c => st.ans[c] && st.ans[c].label).length; }
LABELS.forEach(([k, t], j) => {
  const b = document.createElement("button");
  b.innerHTML = "<kbd>" + (j + 1) + "</kbd>" + t; b.dataset.k = k;
  b.onclick = () => label(k); $("labels").appendChild(b);
});
function render(play) {
  const c = CLIPS[st.i], a = st.ans[c] || {};
  $("count").textContent = (st.i + 1) + " / " + CLIPS.length;
  $("state").textContent = a.label ? "labelled" : "unlabelled";
  $("bar").style.width = (100 * done() / CLIPS.length) + "%";
  document.querySelectorAll("#labels button").forEach(b => b.classList.toggle("sel", b.dataset.k === a.label));
  $("note").value = a.note || "";
  audio.src = "audio/" + c + ".wav";
  $("summary").textContent = done() + " of " + CLIPS.length + " labelled" + (done() === CLIPS.length ? ". Done, download the CSV." : ".");
  if (play) audio.play().catch(() => {});
}
function label(k) {
  const c = CLIPS[st.i];
  st.ans[c] = { label: k, note: $("note").value, t: new Date().toISOString() };
  save();
  const nxt = CLIPS.findIndex((x, j) => j > st.i && !(st.ans[x] && st.ans[x].label));
  if (nxt >= 0) { st.i = nxt; save(); render(true); } else render(false);
}
$("note").addEventListener("input", () => { const c = CLIPS[st.i]; st.ans[c] = Object.assign(st.ans[c] || {}, { note: $("note").value }); save(); });
$("play").onclick = () => { audio.currentTime = 0; audio.play(); };
$("prev").onclick = () => { if (st.i > 0) { st.i--; save(); render(true); } };
$("next").onclick = () => { const n = CLIPS.findIndex(x => !(st.ans[x] && st.ans[x].label)); if (n >= 0) { st.i = n; save(); render(true); } };
$("dl").onclick = () => {
  const esc = s => '"' + String(s || "").replace(/"/g, '""') + '"';
  const rows = ["clip,label,note,time"].concat(CLIPS.map(c => { const a = st.ans[c] || {}; return [c, a.label || "", esc(a.note), a.t || ""].join(","); }));
  const blob = new Blob([rows.join("\\n") + "\\n"], { type: "text/csv" });
  const u = URL.createObjectURL(blob), l = document.createElement("a");
  l.href = u; l.download = "listening_negatives.csv"; l.click(); URL.revokeObjectURL(u);
};
document.addEventListener("keydown", e => {
  if (e.target === $("note")) { if (e.key === "Enter") $("note").blur(); return; }
  if (e.key === " ") { e.preventDefault(); $("play").click(); }
  else if (e.key >= "1" && e.key <= String(LABELS.length)) label(LABELS[+e.key - 1][0]);
  else if (e.key === "ArrowLeft") $("prev").click();
  else if (e.key === "ArrowRight") $("next").click();
});
render(false);
</script>
</body>
</html>
"""


def build(seed: int = 20261022) -> Path:
    meta = pd.read_csv(BANK / "metadata.csv")
    neg = meta[(meta["duration_ms"] == 1000) & (meta["ground_truth"] == "NONSPEECH")].reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(len(neg))
    key = pd.DataFrame({"clip": [f"clip_{k + 1:03d}" for k in range(len(neg))],
                        "base_id": neg["base_id"].to_numpy()[order],
                        "audio_path": neg["audio_path"].to_numpy()[order]})
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    for c, p in zip(key["clip"], key["audio_path"]):
        shutil.copyfile(BANK / "audio" / p, OUT / "audio" / f"{c}.wav")
    key.to_csv(OUT / "key.csv", index=False)
    page = PAGE.replace("__CLIPS__", json.dumps(key["clip"].tolist())).replace("__LABELS__", json.dumps(LABELS))
    (OUT / "index.html").write_text(page, encoding="utf-8")
    return OUT / "index.html"


def read(csv: Path) -> None:
    lab = pd.read_csv(csv)
    key = pd.read_csv(OUT / "key.csv")
    scr = pd.read_csv(WORK / "negatives" / "screen.csv")[["base_id", "p_speech", "p_singing", "p_music", "top3"]]
    t = key.merge(lab, on="clip", how="left").merge(scr, on="base_id", how="left")
    t.to_csv(WORK / "negatives" / "listening.csv", index=False)
    print(t["label"].value_counts(dropna=False).to_string(), "\n")
    t["tagger p >= 0.5"] = (t[["p_speech", "p_singing"]].max(axis=1) >= 0.5)
    print(md(pd.crosstab(t["label"].fillna("no label"), t["tagger p >= 0.5"]), 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--read", type=Path)
    a = ap.parse_args()
    if a.read:
        read(a.read)
    else:
        print(build())


if __name__ == "__main__":
    main()
