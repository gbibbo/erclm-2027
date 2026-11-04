"""Qwen3-Omni as a second model, frozen readouts only (2026-10-29). Exploratory.

Nothing here changes the frozen design. lib_v2 stays the library of Qwen2-Audio and its
readouts, metrics and prompt are reused as they are. This module loads
Qwen3-Omni-30B-A3B-Instruct, thinker only, in NF4 and reads the same native score as
lib_v2, s = z(speech label) - z(non-speech label) at the first answer token. The prompts
are the confirmatory A/B format, the format with the letters swapped that the recipe
averages with it (s_sim), and the prompt of the study under review, which links the
pilot to its generation results.

Two things differ from Qwen2-Audio and are handled here.

- transformers 5 fuses the experts of each MoE layer into 3-D parameters, which
  bitsandbytes does not quantise, so the model would keep about 60 GB in fp16.
  transformers 4.57 keeps one nn.Linear per expert and the thinker fits in NF4 on
  2 x T4 (about 18 GB). The pilot config pins it (configs/omni_pilot.yaml).
- The checkpoint is 70.5 GB, 65 GB of them the thinker. ShardFetcher downloads each
  shard just before transformers loads and quantises it and deletes it afterwards, so
  the disk never holds more than one shard.

A 2 s container gives 26 audio tokens (about 77 ms each), against 50 in Qwen2-Audio. The
encoder convolves the mel frames in chunks of 1 s, each on its own and with its own position
embedding, so an excerpt centred in the container falls on the seam between two chunks
(token_spans). A scoring track can roll the audio by shift_ms to move the excerpt into the
middle of the second chunk, which tests that seam as a cause of the short-excerpt failure.

Every pass also keeps the states the probes of lib_v2 read (forward_scores), so the control
run can tell a poorer representation from a readout that does not use it.

Plan and registered rules in docs/exploratory_log.md, sections "Qwen3-Omni pilot (29 October)" and
"Qwen3-Omni on the 300 control clips (30 October)".
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import lib_v2 as L
from . import lib_v3

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
N_AUDIO_TOKENS_2S = 26

# The prompt of the study under review for Qwen3-Omni (scripts of gbibbo/qwen-vad-lora,
# src/qsm/models/qwen3_omni.py). The study generated text and parsed it. Here the first token
# is read like every other prompt. SPEECH and NONSPEECH start with different tokens.
HAND_CSL = {"system": L.SYSTEM_PROMPT,
            "user": "Does this audio contain human speech? Answer SPEECH or NONSPEECH.",
            "labels": ("SPEECH", "NONSPEECH")}
PROMPTS = {"ab": L.PROMPTS["ab"], "ba_speech_second": lib_v3.PROMPTS["ba_speech_second"],
           "hand_csl": HAND_CSL}
SIM_PAIR = ("ab", "ba_speech_second")
# A quarter, a half and three quarters of the 48 layers, like layers 8, 16 and 24 of 32 in Qwen2-Audio
LLM_LAYERS = (12, 24, 36)


def as_track(t) -> dict:
    """A scoring track, a prompt read on the audio rolled by shift_ms, on some conditions or all."""
    t = {"name": t, "prompt": t} if isinstance(t, str) else dict(t)
    t.setdefault("name", t["prompt"])
    t["shift_ms"] = float(t.get("shift_ms") or 0.0)
    t["conditions"] = list(t["conditions"]) if t.get("conditions") else None
    if t["prompt"] not in PROMPTS:
        raise ValueError(f"track {t['name']}: unknown prompt {t['prompt']}")
    return t


def tracks(o: dict) -> list:
    """The tracks of the omni block of a config. With only `prompts`, one track per prompt."""
    ts = [as_track(t) for t in (o.get("tracks") or o["prompts"])]
    names = [t["name"] for t in ts]
    if len(set(names)) != len(names):
        raise ValueError(f"repeated track names {names}")
    for t in ts:
        extra = set(t["conditions"] or []) - set(o["conditions"])
        if extra:
            raise ValueError(f"track {t['name']}: conditions {sorted(extra)} are not in omni.conditions")
    return ts


def _conv_len(n: int) -> int:
    """Tokens left by the three stride-2 convolutions of the audio encoder from n mel frames."""
    for _ in range(3):
        n = (n - 1) // 2 + 1
    return n


def token_spans(n_frames: int, n_window: int, hop_ms: float = 1000.0 * L.HOP / L.SR) -> np.ndarray:
    """Start and end in ms of the audio each encoder token covers, as an (n_tokens, 2) array.

    The encoder cuts the mel frames into chunks of 2 * n_window frames (100, so 1 s), convolves
    each chunk on its own and restarts the position embedding in each. Token j of a chunk covers
    frames 8j to 8j + 7 of it, cut at the chunk end. A 2 s container gives two chunks of 13
    tokens, and the excerpt centred in it falls on the seam between them.
    """
    chunk, spans = 2 * int(n_window), []
    for c0 in range(0, int(n_frames), chunk):
        n = min(chunk, int(n_frames) - c0)
        for j in range(_conv_len(n)):
            spans.append(((c0 + 8 * j) * hop_ms, (c0 + min(8 * j + 8, n)) * hop_ms))
    return np.array(spans, dtype=float).reshape(-1, 2)


def region_indices(spans: np.ndarray, segment_ms: float, container_ms: float, shift_ms: float = 0.0,
                   margin: int = 1) -> np.ndarray:
    """Tokens whose span overlaps the excerpt, plus `margin` tokens on each side, as lib_v2.

    The excerpt is centred in the container and moved by shift_ms when its track rolls the audio.
    """
    s0 = (container_ms - float(segment_ms)) / 2.0 + shift_ms
    s1 = s0 + float(segment_ms)
    if s0 < 0 or s1 > container_ms + 1e-6:
        raise ValueError(f"an excerpt of {segment_ms} ms shifted by {shift_ms} ms leaves the container")
    hit = np.flatnonzero((spans[:, 0] < s1) & (spans[:, 1] > s0))
    return np.arange(max(hit[0] - margin, 0), min(hit[-1] + margin, len(spans) - 1) + 1)


# ============================================================================ loading

def disk_report(paths) -> dict:
    out = {}
    for p in paths:
        try:
            u = shutil.disk_usage(p)
            out[str(p)] = {"free_gb": round(u.free / 1e9, 1), "total_gb": round(u.total / 1e9, 1)}
        except OSError:
            pass
    return out


class ShardFetcher:
    """Downloads each checkpoint shard right before transformers 4.57 loads it, then deletes it.

    The local folder holds the small files and the index. Every shard listed in the index
    starts as an empty placeholder, which is enough for the loader to resolve the files.
    transformers.modeling_utils.load_shard_file is wrapped while the model loads, so the
    real shard is fetched, quantised onto the GPU and removed one at a time.
    """

    def __init__(self, local_dir, fetch, log=print, after_each=None):
        self.dir = Path(local_dir)
        self.fetch = fetch          # fetch(filename, dest_path) puts the real shard at dest_path
        self.log = log
        self.after_each = after_each  # called after each shard, returns a note for the log
        self.loaded = []

    def prepare(self, index: dict):
        shards = sorted(set(index["weight_map"].values()))
        for name in shards:
            p = self.dir / name
            if not p.exists() or p.stat().st_size == 0:
                p.write_bytes(b"")
        return shards

    def wrap(self, load_shard_file):
        def wrapped(args):
            path = Path(args[0])
            if args[0] and path.resolve().parent == self.dir.resolve():
                t0 = time.time()
                path.unlink(missing_ok=True)
                self.fetch(path.name, path)
                size = path.stat().st_size
                try:
                    return load_shard_file(args)
                finally:
                    path.unlink(missing_ok=True)
                    path.write_bytes(b"")
                    self.loaded.append(path.name)
                    note = self.after_each() if self.after_each else ""
                    self.log(f"shard {path.name}: {size / 1e9:.2f} GB fetched and loaded in "
                             f"{time.time() - t0:.0f} s{note}")
            return load_shard_file(args)
        return wrapped


def hub_fetcher(repo_id: str, revision: str, token: Optional[str]):
    def fetch(filename, dest):
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id, filename, revision=revision, token=token, local_dir=str(Path(dest).parent))
        if Path(p) != Path(dest):
            shutil.move(p, dest)
    return fetch


def thinker_only_index(index: dict) -> dict:
    """The index restricted to the thinker, so the talker shards are never fetched."""
    wm = {k: v for k, v in index["weight_map"].items() if k.startswith("thinker.")}
    return {"metadata": index.get("metadata", {}), "weight_map": wm}


def release_cache() -> str:
    """Returns the cached blocks to the GPU after a shard and reports allocated/reserved GB per GPU.

    The first pilot attempt (omni-pilot-v1) ran out of memory while quantising, with 8.2 GiB
    reserved but unallocated on GPU 1. Quantising leaves many small blocks that the default
    allocator cannot merge. The config also sets PYTORCH_CUDA_ALLOC_CONF to expandable segments.
    """
    import gc
    import torch
    gc.collect()
    if not torch.cuda.is_available():
        return ""
    torch.cuda.empty_cache()
    gb = [f"{torch.cuda.memory_allocated(i) / 2**30:.1f}/{torch.cuda.memory_reserved(i) / 2**30:.1f}"
          for i in range(torch.cuda.device_count())]
    return f", GPU allocated/reserved GiB {gb}"


def compute_dtype(name: str):
    import torch
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load(cfg: dict, hf_token: Optional[str] = None, work=None, log=print):
    """Qwen3-Omni thinker in NF4 with double quantisation, at the pinned revision.

    Returns (model, processor, info). Fails when part of the model would sit on the CPU
    or the disk, since that changes the computation and the speed.
    """
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, BitsAndBytesConfig, Qwen3OmniMoeForConditionalGeneration
    from transformers import Qwen3OmniMoeProcessor
    m, o = cfg["model"], cfg.get("omni", {})
    work = Path(work or ".")
    local = work / "models" / m["id"].split("/")[-1]
    local.mkdir(parents=True, exist_ok=True)
    info = {"transformers": transformers.__version__, "torch": torch.__version__,
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "disk_before": disk_report([work, "/tmp", Path.home()])}
    log(f"transformers {transformers.__version__}, disk {info['disk_before']}")
    small = ["config.json", "generation_config.json", "chat_template.json", "preprocessor_config.json",
             "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors.index.json"]
    for f in small:
        hf_hub_download(m["id"], f, revision=m["revision"], token=hf_token, local_dir=str(local))
    index = json.loads((local / "model.safetensors.index.json").read_text())
    index = thinker_only_index(index)
    (local / "model.safetensors.index.json").write_text(json.dumps(index))
    fetcher = ShardFetcher(local, hub_fetcher(m["id"], m["revision"], hf_token), log=log, after_each=release_cache)
    shards = fetcher.prepare(index)
    info["shards"] = shards

    config = AutoConfig.from_pretrained(str(local))
    config.enable_audio_output = False  # thinker only, the talker is never built
    dt = compute_dtype(o.get("compute_dtype", "float16"))
    if m["quant"] != "nf4":
        raise ValueError("the Qwen3-Omni pilot is defined in NF4")
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt,
                            bnb_4bit_use_double_quant=True)
    n = torch.cuda.device_count()
    max_memory = {i: o.get("max_memory_per_gpu", "13GiB") for i in range(n)} or None
    import transformers.modeling_utils as mu
    original = mu.load_shard_file
    mu.load_shard_file = fetcher.wrap(original)
    t0 = time.time()
    try:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            str(local), config=config, quantization_config=qc, dtype=dt, device_map="auto",
            max_memory=max_memory).eval()
    finally:
        mu.load_shard_file = original
    if sorted(fetcher.loaded) != shards:
        raise RuntimeError(f"loaded shards {sorted(fetcher.loaded)} differ from the index {shards}")
    placement = sorted({str(v) for v in (getattr(model, "hf_device_map", None) or {}).values()})
    if any(p in ("cpu", "disk") for p in placement):
        raise RuntimeError(f"part of the model was placed on {placement}. It needs 2 GPUs of about 15 GB")
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(local))
    info.update(load_s=round(time.time() - t0, 1), placement=placement, compute_dtype=str(dt),
                disk_after=disk_report([work, "/tmp"]))
    return model, processor, info


# ============================================================================ readout

def thinker(model):
    return getattr(model, "thinker", model)


def build_text(processor, prompt: dict) -> str:
    return L.build_text(processor, prompt)


def forward_scores(model, processor, text: str, wav: np.ndarray, label_info: dict, audio_token_id: int,
                   dtype=None, check_head: bool = False, segment_ms: Optional[float] = None,
                   shift_ms: float = 0.0, layers=LLM_LAYERS):
    """One prefill pass. Returns (scalars, float16 feature vectors).

    h_final is the last hidden state at the answer position, after the final norm, so
    lm_head(h_final) reproduces the logits (checked on the first items). h_layer<k> is the
    output of layer k at that position. The encoder states are those of lib_v2. enc_* come
    before the output projection (ln_post, 1280 dims), averaged over the container or, given
    segment_ms, averaged and max-pooled over the excerpt tokens with one token of margin.
    proj_region_mean is the projected state the language model reads (2048 dims).
    """
    import torch
    th = thinker(model)
    inputs = L.prepare_inputs(processor, text, wav)
    fam = inputs.get("feature_attention_mask")
    n_frames = int(fam[0].sum()) if fam is not None else int(inputs["input_features"].shape[-1])
    dev = next(th.parameters()).device
    dt = dtype or th.lm_head.weight.dtype
    inputs = {k: (v.to(dev, dtype=dt) if v.is_floating_point() else v.to(dev)) for k, v in inputs.items()}
    tower, rec = th.audio_tower, {}
    hooks = [tower.ln_post.register_forward_hook(lambda m, i, o: rec.__setitem__("enc", o)),
             tower.register_forward_hook(
                 lambda m, i, o: rec.__setitem__("proj", o[0] if isinstance(o, tuple) else o.last_hidden_state))]
    try:
        with torch.inference_mode():
            out = th(**inputs, use_cache=False, output_hidden_states=True)
            logits = out.logits[0, -1].float()
            h = out.hidden_states[-1][0, -1]
            head_diff = None
            if check_head:
                w = th.lm_head.weight
                head = th.lm_head(h.to(device=w.device, dtype=w.dtype)).float()
                ids = [label_info["pos_primary"], label_info["neg_primary"]]
                head_diff = float((head[ids].to(logits.device) - logits[ids]).abs().max())
    finally:
        for hk in hooks:
            hk.remove()
    pos, neg = label_info["pos_ids"], label_info["neg_ids"]
    zA, zB = float(logits[label_info["pos_primary"]]), float(logits[label_info["neg_primary"]])
    top1 = int(torch.argmax(logits))
    scal = {"zA": zA, "zB": zB, "s_primary": zA - zB,
            "s_lse": float(torch.logsumexp(logits[pos], 0) - torch.logsumexp(logits[neg], 0)),
            "top1": top1, "top1_is_label": bool(top1 in pos or top1 in neg),
            "n_audio_in_ids": int((inputs["input_ids"][0] == audio_token_id).sum())}
    if head_diff is not None:
        scal["head_check_maxdiff"] = head_diff
    spans = token_spans(n_frames, tower.n_window)
    enc, proj = rec["enc"].float(), rec["proj"].float()
    if not enc.shape[0] == proj.shape[0] == len(spans) == scal["n_audio_in_ids"]:
        raise RuntimeError(f"audio tokens differ: encoder {enc.shape[0]}, projection {proj.shape[0]}, "
                           f"chunk geometry {len(spans)}, prompt {scal['n_audio_in_ids']}")
    feats = {"h_final": h, "enc_global_mean": enc.mean(0)}
    for k in layers:
        if k + 1 < len(out.hidden_states) - 1:
            feats[f"h_layer{k}"] = out.hidden_states[k + 1][0, -1]
    if segment_ms is not None:
        idx = region_indices(spans, segment_ms, n_frames * 1000.0 * L.HOP / L.SR, shift_ms)
        feats.update(enc_region_mean=enc[idx].mean(0), enc_region_max=enc[idx].max(0).values,
                     proj_region_mean=proj[idx].mean(0))
        scal.update(region_k0=int(idx[0]), region_k1=int(idx[-1]))
    return scal, {k: v.float().cpu().numpy().astype(np.float16) for k, v in feats.items()}


def check(model, processor, prompt_key: str = "ab", head_tol: float = 0.01) -> dict:
    """The model checks of the runner for Qwen3-Omni, on the 2 s probe of the runner."""
    from .runner import probe_wav
    tok = processor.tokenizer
    labels = L.label_token_info(tok, PROMPTS[prompt_key]["labels"])
    aid = tok.convert_tokens_to_ids("<|audio_pad|>")
    text = build_text(processor, PROMPTS[prompt_key])
    wav = probe_wav()
    s1, a1 = forward_scores(model, processor, text, wav, labels, aid, check_head=True)
    s2, a2 = forward_scores(model, processor, text, wav, labels, aid)
    res = {**s1, "deterministic": s1["s_primary"] == s2["s_primary"],
           "top1_decoded": tok.decode([s1["top1"]]),
           "labels": {**labels, "decoded": [tok.decode([i]) for i in labels["pos_ids"] + labels["neg_ids"]]},
           "h_final_dim": int(a1["h_final"].shape[0]), "enc_dim": int(a1["enc_global_mean"].shape[0])}
    failures = []
    if not np.isfinite([s1["zA"], s1["zB"]]).all() or not all(np.isfinite(v.astype(float)).all() for v in a1.values()):
        failures.append(f"non-finite logits or state (zA {s1['zA']}, zB {s1['zB']}), check the compute dtype")
    if s1["n_audio_in_ids"] != N_AUDIO_TOKENS_2S:
        failures.append(f"a 2 s WAV gave {s1['n_audio_in_ids']} audio tokens, expected {N_AUDIO_TOKENS_2S}")
    if not s1["head_check_maxdiff"] < head_tol:
        failures.append(f"the final state does not reproduce the logits, max diff {s1['head_check_maxdiff']:.3g}")
    res["failures"] = failures
    return res


def run(model, processor, manifest: pd.DataFrame, store, track, on_item=None,
        n_head_checks: int = 3, log_every: int = 50, log=print, layers=LLM_LAYERS):
    """Scores and probe states for every manifest row not yet in the store, with one track.

    track is a prompt key or a dict of as_track. With shift_ms the audio is rolled by that
    much, so the excerpt moves and the container wraps around (noise or near silence).
    """
    t = as_track(track)
    tok = processor.tokenizer
    labels = L.label_token_info(tok, PROMPTS[t["prompt"]]["labels"])
    aid = tok.convert_tokens_to_ids("<|audio_pad|>")
    text = build_text(processor, PROMPTS[t["prompt"]])
    shift = int(round(t["shift_ms"] * L.SR / 1000.0))
    done = store.done_ids()
    todo = manifest[~manifest["variant_id"].astype(str).isin(done)]
    t0, n = time.time(), len(todo)
    log(f"[{t['name']}] {len(done)} items already stored, {n} to score")
    for i, row in enumerate(todo.itertuples(index=False)):
        wav = L.load_wav(row.path)
        if shift:
            wav = np.roll(wav, shift)
        scal, arrays = forward_scores(model, processor, text, wav, labels, aid, check_head=i < n_head_checks,
                                      segment_ms=float(row.duration_ms), shift_ms=t["shift_ms"], layers=layers)
        meta = {c: getattr(row, c) for c in L.META_COLS}
        meta.update(scal)
        meta.update(prompt=t["prompt"], track=t["name"], shift_ms=t["shift_ms"])
        store.add(meta, arrays)
        if on_item:
            on_item()
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            log(f"  [{t['name']}] {i + 1}/{n}  {el / (i + 1):.2f} s/item  "
                f"ETA {(n - i - 1) * el / (i + 1) / 60:.1f} min")
    store.flush()
    return {"n_scored": n, "s_per_item": round((time.time() - t0) / n, 3) if n else None}


# ============================================================================ pilot analysis

def with_sim(metas: dict) -> dict:
    """Adds s_sim, the mean of s over the two A/B formats, when both are present."""
    out = dict(metas)
    a, b = SIM_PAIR
    if a in metas and b in metas:
        m = metas[a].merge(metas[b][["variant_id", "s_primary"]], on="variant_id", suffixes=("", "_b"))
        if len(m) != len(metas[a]):
            raise RuntimeError(f"s_sim: {len(metas[a]) - len(m)} items of {a} are missing in {b}")
        m["s_primary"] = 0.5 * (m["s_primary"] + m.pop("s_primary_b"))
        m["prompt"] = "s_sim"
        out["s_sim"] = m
    return out


def _metrics(meta: pd.DataFrame, w=None) -> dict:
    y = meta["label"].to_numpy().astype(bool)
    s = meta["s_primary"].to_numpy(dtype=float)
    thr = L.apply_rule(meta, s, "bc_cond", w=w)
    return {"auroc": L.auc_w(y, s, w), "ba_L0": L.ba_w(y, s > 0, w), "ba_L1": L.ba_w(y, s > thr, w),
            "oracle_ba": L.oracle_ba_w(y, s, w)}


def condition_table(metas: dict, B: int = 2000, seed: int = 42) -> pd.DataFrame:
    """BA (native rule and Batch Calibration per condition), best-threshold BA and AUROC per
    prompt and condition, with 95% percentile intervals from a bootstrap over recordings,
    stratified by class. The calibration is re-estimated in every replicate."""
    rows = []
    for p, meta in with_sim(metas).items():
        for cond, m in meta.groupby("condition", sort=False):
            m = m.reset_index(drop=True)
            point = _metrics(m)
            y = m["label"].to_numpy().astype(bool)
            s = m["s_primary"].to_numpy(dtype=float)
            sdt = L.sdt_from_pred(y, s > 0)
            rng = np.random.default_rng(seed)
            recs = m["rec_id"].astype(str).to_numpy()
            by_class = [np.unique(recs[y == c]) for c in (True, False)]
            reps = []
            for _ in range(B):
                cnt = {}
                for u in by_class:
                    for r in rng.choice(u, size=len(u), replace=True):
                        cnt[r] = cnt.get(r, 0) + 1
                w = np.array([cnt.get(r, 0) for r in recs], dtype=float)
                reps.append(_metrics(m, w))
            reps = pd.DataFrame(reps)
            row = {"prompt": p, "condition": cond, "n": len(m), **point,
                   "H": sdt["H"], "F": sdt["F"],
                   "top1_is_label": float(m["top1_is_label"].mean()) if "top1_is_label" in m else np.nan}
            for k in ("auroc", "ba_L0", "ba_L1"):
                row[f"{k}_lo"], row[f"{k}_hi"] = (float(np.nanpercentile(reps[k], 2.5)),
                                                  float(np.nanpercentile(reps[k], 97.5)))
            rows.append(row)
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame, s_per_item: Optional[float], rule: dict) -> dict:
    """The registered reading of the pilot (docs/exploratory_log.md, "Qwen3-Omni pilot (29 October)")."""
    t = table.set_index(["prompt", "condition"])
    out = {"rule": rule}
    vp, vc = rule["validity_prompt"], rule["validity_condition"]
    v = float(t.loc[(vp, vc), "ba_L0"]) if (vp, vc) in t.index else float("nan")
    out["validity"] = {"ba_L0": v, "min": rule["validity_min_ba"], "ok": bool(v >= rule["validity_min_ba"])}
    if s_per_item:
        hours = s_per_item * rule["full_items"] / 3600
        out["feasibility"] = {"s_per_item": s_per_item, "full_items": rule["full_items"],
                              "projected_h": round(hours, 1), "max_h": rule["max_hours"],
                              "ok": bool(hours <= rule["max_hours"])}
    reading = {}
    for p in rule["read_prompts"]:
        for c in rule["read_conditions"]:
            if (p, c) not in t.index:
                continue
            r = t.loc[(p, c)]
            if r["auroc_hi"] < rule["representation_auroc_hi"]:
                kind = "representation"
            elif r["auroc_lo"] > rule["criterion_auroc_lo"] and r["oracle_ba"] - r["ba_L0"] >= rule["criterion_gap"]:
                kind = "criterion"
            else:
                kind = "inconclusive"
            reading[f"{p}/{c}"] = {"reading": kind, "auroc": float(r["auroc"]),
                                   "auroc_ci": [float(r["auroc_lo"]), float(r["auroc_hi"])],
                                   "ba_L0": float(r["ba_L0"]), "oracle_ba": float(r["oracle_ba"]),
                                   "ba_L1": float(r["ba_L1"])}
    out["reading"] = reading
    out["full_run"] = bool(out["validity"]["ok"] and out.get("feasibility", {}).get("ok", False))
    return out
