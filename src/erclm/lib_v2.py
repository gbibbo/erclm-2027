"""ERC-LM pilot library, v2 (2026-10-15).

Changes from v1
  - Label-free thresholds (Batch Calibration, centring, median) are recomputed
    inside every bootstrap replicate from the resampled, unlabeled batch.
    Readouts are (score, threshold, rule) triples; rule=None means fixed.
  - Heads fitted by cross-fitting on the evaluation set are named L2_cv and
    P_<feature>_cv (exploratory). Heads fitted on separate supervision keep
    the names L2 and P_<feature> (confirmatory).
  - combine_runs() aligns Base and LoRA runs on shared items, so H3 and H4 are
    estimated with one paired bootstrap over the same recordings.
  - Contrasts report the bootstrap SE and an 80%-power minimum detectable
    effect. prevalence_check() probes the 50/50 assumption of batch rules.

Extraction and analysis for the ERC-LM 2027 study on how much voice activity
detection (VAD) performance of Qwen2-Audio can be recovered by label-free
criterion calibration (L1), a supervised linear readout on frozen states (L2)
and LoRA fine-tuning, across controlled duration and SNR degradations.

Sections
  A. Manifest     condition parsing from the file names of the study under review,
                  ids, pilot sampling, content-free containers
  B. Geometry     audio-token indices covering the centred segment
  C. Extraction   prompts, label tokens, forward hooks, LoRA loading and
                  verification, resumable feature store
  D. Readouts     native score (L0), label-free calibration (L1), linear heads
                  with grouped cross-fitting (L2) and encoder probes
  E. Metrics      weighted AUROC and BA, oracle BA (L1*), d_a, relative
                  decoding efficiency, SDT d' and c, per-condition table
  F. Bootstrap    paired cluster bootstrap by source recording, contrasts
  G. Thresholds   sustained crossing with censoring and ambiguity flags

torch, transformers, peft and soundfile are imported lazily, so sections A, B
and D to G run on CPU-only machines.
"""
from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

__version__ = "2.0.0"

SR = 16000
HOP = 160
CONTAINER_MS = 2000.0
PAD_SIGMA = 1e-4

# ============================================================================
# A. Manifest
# ============================================================================

_RE_DUR = re.compile(r"_dur(\d+(?:\.\d+)?)ms")
_RE_SNR = re.compile(r"_snr([+-]?\d+(?:\.\d+)?)dB")
_RE_REV = re.compile(r"_reverb(none|\d+(?:\.\d+)?s)")
_RE_FIL = re.compile(r"_filter(none|bandpass|lowpass|highpass)")
_RE_VOX = re.compile(r"voxconverse_([A-Za-z0-9]+)_")
_RE_ESC = re.compile(r"esc50_(\d+)-(\d+)-[A-Za-z]+-\d+_")
_RE_BASE = re.compile(r"^(.*?)_\d+ms_(?:dur|snr|reverb|filter)")

META_COLS = ["variant_id", "base_id", "rec_id", "label", "role", "axis", "condition",
             "duration_ms", "snr_db", "t60_s", "reverb", "filter"]


def _num(x: float) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:g}"


def parse_variant(name: str) -> dict:
    """Degradation factors from a file name in the scheme of the study under review.

    Handles one-factor test variants (..._1000ms_dur20ms.wav, ..._snr+0dB.wav,
    ..._reverb0.3s.wav, ..._filterbandpass.wav) and factorial training variants
    (..._1000ms_dur100ms_snr-10dB.wav). Absent factors take neutral values
    (1000 ms, no added noise, dry, unfiltered).
    """
    stem = Path(str(name)).name
    f = {"duration_ms": 1000.0, "snr_db": np.nan, "t60_s": 0.0,
         "reverb": "none", "filter": "none"}
    present = []
    m = _RE_DUR.search(stem)
    if m:
        f["duration_ms"] = float(m.group(1))
        present.append("duration")
    m = _RE_SNR.search(stem)
    if m:
        f["snr_db"] = float(m.group(1))
        present.append("snr")
    m = _RE_REV.search(stem)
    if m:
        f["reverb"] = m.group(1)
        f["t60_s"] = 0.0 if m.group(1) == "none" else float(m.group(1)[:-1])
        present.append("reverb")
    m = _RE_FIL.search(stem)
    if m:
        f["filter"] = m.group(1)
        present.append("filter")
    if len(present) == 1:
        f["axis"] = present[0]
    elif len(present) > 1:
        f["axis"] = "factorial"
    else:
        f["axis"] = "unknown"
    parts = []
    if "duration" in present:
        parts.append(f"dur_{_num(f['duration_ms'])}ms")
    if "snr" in present:
        parts.append(f"snr_{_num(f['snr_db'])}dB")
    if "reverb" in present:
        parts.append(f"reverb_{f['reverb']}")
    if "filter" in present:
        parts.append(f"filter_{f['filter']}")
    f["condition"] = "|".join(parts) if parts else "unknown"
    return f


def base_id(name: str) -> str:
    """Base clip id: the file stem before the variant suffix."""
    stem = Path(str(name)).stem
    m = _RE_BASE.match(stem)
    return m.group(1) if m else stem


def recording_id(name: str) -> str:
    """Source recording, the independent unit for resampling.

    voxconverse_wewoz_0454_... -> vox:wewoz
    esc50_1-45645-A-31_0152_... -> esc:1-45645 (fold and freesound source; takes
    A, B, ... of one source share a recording)
    """
    stem = Path(str(name)).name
    m = _RE_VOX.match(stem)
    if m:
        return "vox:" + m.group(1)
    m = _RE_ESC.match(stem)
    if m:
        return f"esc:{m.group(1)}-{m.group(2)}"
    return "other:" + base_id(stem)


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"none of {list(candidates)} in columns {list(df.columns)}")


def _to_binary(labels: pd.Series) -> np.ndarray:
    out = []
    for v in labels:
        s = str(v).strip().upper()
        if s in ("SPEECH", "1", "TRUE", "A"):
            out.append(1)
        elif s in ("NONSPEECH", "NON-SPEECH", "0", "FALSE", "B"):
            out.append(0)
        else:
            raise ValueError(f"unknown label {v!r}")
    return np.asarray(out, dtype=int)


def load_manifest(csv_path, audio_root=None, role: str = "test",
                  path_col: Optional[str] = None, label_col: Optional[str] = None) -> pd.DataFrame:
    """Standardised manifest from any CSV with an audio-path and a label column.

    audio_root: if given, each file is looked up as audio_root/<file name>. Use it
    when the CSV holds absolute paths from another machine.
    """
    raw = pd.read_csv(csv_path)
    pcol = path_col or _find_col(raw, ("audio_path", "path", "filepath", "file", "wav_path"))
    lcol = label_col or _find_col(raw, ("ground_truth", "label", "y"))
    names = raw[pcol].astype(str).tolist()
    parsed = pd.DataFrame([parse_variant(n) for n in names])
    df = pd.DataFrame({
        "variant_id": [Path(n).stem for n in names],
        "base_id": [base_id(n) for n in names],
        "rec_id": [recording_id(n) for n in names],
        "label": _to_binary(raw[lcol]),
        "role": role,
    })
    df = pd.concat([df, parsed], axis=1)
    df["path"] = [str(Path(audio_root) / Path(n).name) if audio_root else n for n in names]
    if df["variant_id"].duplicated().any():
        dup = df.loc[df["variant_id"].duplicated(), "variant_id"].iloc[0]
        raise ValueError(f"duplicated variant_id, e.g. {dup}")
    return df


def select_pilot(df: pd.DataFrame, n_per_class: int = 50, seed: int = 42,
                 one_per_recording: bool = True) -> pd.DataFrame:
    """All variants of n_per_class base clips per class, at most one clip per recording."""
    bases = (df.drop_duplicates("base_id")[["base_id", "rec_id", "label"]]
             .sort_values("base_id").reset_index(drop=True))
    rng = np.random.default_rng(seed)
    chosen = []
    for lab in (1, 0):
        b = bases[bases["label"] == lab]
        b = b.iloc[rng.permutation(len(b))]
        if one_per_recording:
            b = b.drop_duplicates("rec_id")
        if len(b) < n_per_class:
            raise ValueError(f"only {len(b)} eligible base clips for label {lab}")
        chosen.extend(b["base_id"].iloc[:n_per_class].tolist())
    return df[df["base_id"].isin(chosen)].reset_index(drop=True)


def make_content_free(out_dir, n_noise_seeds: int = 8, container_ms: float = CONTAINER_MS,
                      sigma: float = PAD_SIGMA, sr: int = SR) -> pd.DataFrame:
    """Speech-free containers for the content-free calibration variant (L1_cf).

    Writes n_noise_seeds containers of Gaussian padding noise (the padding of the study under review)
    plus one of digital silence, as float WAV, and returns their manifest rows.
    """
    import soundfile as sf
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = int(round(container_ms * sr / 1000.0))
    signals = {f"cf_pad_seed{k:02d}": np.random.default_rng(10_000 + k).standard_normal(n) * sigma
               for k in range(n_noise_seeds)}
    signals["cf_silence"] = np.zeros(n)
    rows = []
    for name, x in signals.items():
        p = out / f"{name}.wav"
        sf.write(str(p), x.astype(np.float32), sr, subtype="FLOAT")
        rows.append({"variant_id": name, "base_id": name, "rec_id": "cf", "label": 0,
                     "role": "content_free", "duration_ms": 0.0, "snr_db": np.nan, "t60_s": 0.0,
                     "reverb": "none", "filter": "none", "axis": "content_free",
                     "condition": "cf_silence" if name == "cf_silence" else "cf_pad",
                     "path": str(p)})
    return pd.DataFrame(rows)


# ============================================================================
# B. Geometry
# ============================================================================

def n_audio_tokens(n_samples: int, hop: int = HOP) -> int:
    """Audio tokens Qwen2-Audio hands to the LLM for n_samples of 16 kHz audio.

    Mirrors the processor: valid mel frames = ceil(n/hop), conv stride 2, then
    average pooling stride 2. 2.0 s gives 50 tokens of 40 ms.
    """
    mel = -(-int(n_samples) // hop)
    l1 = (mel - 1) // 2 + 1
    return (l1 - 2) // 2 + 1


def region_token_indices(segment_ms: float, n_tokens: int, container_ms: float = CONTAINER_MS,
                         margin: int = 1) -> np.ndarray:
    """Indices of audio tokens overlapping the centred segment, plus a margin."""
    dt = container_ms / n_tokens
    s0 = (container_ms - float(segment_ms)) / 2.0
    s1 = s0 + float(segment_ms)
    k0 = int(math.floor(s0 / dt + 1e-9)) - margin
    k1 = int(math.ceil(s1 / dt - 1e-9)) - 1 + margin
    k0, k1 = max(k0, 0), min(k1, n_tokens - 1)
    if k1 < k0:
        k1 = k0
    return np.arange(k0, k1 + 1)


# ============================================================================
# C. Extraction
# ============================================================================

SYSTEM_PROMPT = "You classify audio content."
PROMPTS = {
    # Format the LoRA of the study under review was trained on. Shared by both backbones (protocol L0).
    "ab": {"system": SYSTEM_PROMPT,
           "user": ("Choose one:\nA) SPEECH (human voice)\nB) NONSPEECH (music/noise/silence/animals)\n\n"
                    "Answer with A or B ONLY."),
           "labels": ("A", "B")},
    # Best prompt of the study under review for the LoRA backbone (OPRO-Template). Sensitivity analysis only.
    "opro_tmpl": {"system": SYSTEM_PROMPT,
                  "user": ("Detect human speech. Treat the following as NONSPEECH: pure tones/beeps, clicks, "
                           "clock ticks, music, environmental noise, silence.\nAnswer: SPEECH or NONSPEECH."),
                  "labels": ("SPEECH", "NONSPEECH")},
}


def build_text(processor, prompt: dict) -> str:
    conv = [{"role": "system", "content": prompt["system"]},
            {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": prompt["user"]}]}]
    return processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


def label_token_info(tokenizer, labels: Sequence[str]) -> dict:
    """First-token ids for the two answer labels.

    The native score is the difference of first-token logits. It identifies a
    forced choice between the two label strings as long as their first tokens
    differ, even when a label spans several tokens. Variants with and without a
    leading space are pooled per class; ids shared by both classes are dropped.
    """
    sets, primary, single = [], [], []
    for lab in labels:
        ids = set()
        for v in (lab, " " + lab):
            enc = tokenizer.encode(v, add_special_tokens=False)
            if enc:
                ids.add(int(enc[0]))
        enc0 = tokenizer.encode(lab, add_special_tokens=False)
        primary.append(int(enc0[0]))
        single.append(len(enc0) == 1)
        sets.append(ids)
    shared = sets[0] & sets[1]
    pos, neg = sorted(sets[0] - shared), sorted(sets[1] - shared)
    if primary[0] == primary[1] or primary[0] not in pos or primary[1] not in neg:
        raise ValueError(f"labels {labels} share their first token; first-token readout not identifiable")
    return {"labels": list(labels), "pos_ids": pos, "neg_ids": neg,
            "pos_primary": primary[0], "neg_primary": primary[1], "single_token": single}


def prepare_inputs(processor, text: str, wav: np.ndarray, sr: int = SR):
    try:
        return processor(text=text, audio=[wav], sampling_rate=sr, return_tensors="pt", padding=True)
    except TypeError:
        return processor(text=text, audios=[wav], sampling_rate=sr, return_tensors="pt", padding=True)


def load_wav(path, sr: int = SR) -> np.ndarray:
    import soundfile as sf
    x, s = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if s != sr:
        import librosa
        x = librosa.resample(x, orig_sr=s, target_sr=sr)
    return np.ascontiguousarray(x, dtype=np.float32)


def _find_module(model, pattern: str):
    rx = re.compile(pattern)
    hits = [(n, m) for n, m in model.named_modules() if rx.search(n)]
    if len(hits) != 1:
        raise RuntimeError(f"pattern {pattern!r} matched {len(hits)} modules: {[h[0] for h in hits][:5]}")
    return hits[0]


def _first_tensor(out):
    import torch
    if isinstance(out, torch.Tensor):
        return out
    if getattr(out, "last_hidden_state", None) is not None:
        return out.last_hidden_state
    if isinstance(out, (tuple, list)) and len(out) and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"cannot extract a tensor from {type(out)}")


class Recorder:
    """Forward hooks that capture the states the protocol reads.

    final_norm   LLM state after the final norm at the last prompt position. The
                 unembedding reads exactly this vector (S4 and S5).
    audio_tower  encoder output after pooling, all positions (S1).
    projector    projector output, all positions.
    layer{L}     residual stream after decoder layer L, last position.
    Module names are matched by pattern, so the same code works with the
    transformers 4.x and 5.x layouts and inside a PEFT wrapper.
    """

    PAT_NORM = r"language_model(?:\.model)?\.norm$"
    PAT_LAYER = r"language_model(?:\.model)?\.layers\.(\d+)$"
    PAT_AUDIO = r"(?:^|\.)audio_tower$"
    PAT_PROJ = r"(?:^|\.)multi_modal_projector$"
    PAT_HEAD = r"(?:^|\.)lm_head$"

    def __init__(self, model, llm_layers: Iterable[int] = ()):
        self.data, self.handles, self.names = {}, [], {}
        for key, pat, last in (("final_norm", self.PAT_NORM, True),
                               ("audio_tower", self.PAT_AUDIO, False),
                               ("projector", self.PAT_PROJ, False)):
            name, mod = _find_module(model, pat)
            self.names[key] = name
            self._hook(mod, key, last)
        self.names["lm_head"], self.lm_head = _find_module(model, self.PAT_HEAD)
        rx = re.compile(self.PAT_LAYER)
        layers = {}
        for name, mod in model.named_modules():
            mm = rx.search(name)
            if mm:
                layers[int(mm.group(1))] = (name, mod)
        self.n_llm_layers = len(layers)
        for L in llm_layers:
            Lr = L if L >= 0 else self.n_llm_layers + L
            name, mod = layers[Lr]
            self.names[f"layer{Lr}"] = name
            self._hook(mod, f"layer{Lr}", True)

    def _hook(self, module, key: str, last: bool):
        def fn(mod, inp, out):
            t = _first_tensor(out).detach()
            self.data[key] = t[:, -1, :] if last else t
        self.handles.append(module.register_forward_hook(fn))

    def clear(self):
        self.data = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def model_device(model):
    return next(model.parameters()).device


def forward_collect(model, recorder: Recorder, inputs, label_info: dict, segment_ms: float,
                    n_samples: int, audio_token_id: Optional[int] = None, sr: int = SR,
                    check_head: bool = False):
    """One forward pass. Returns (scalars, float16 feature vectors)."""
    import torch
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("forward_collect expects batch size 1")
    recorder.clear()
    with torch.inference_mode():
        out = model(**inputs, use_cache=False)
    logits = out.logits[0, -1].float()
    pos, neg = label_info["pos_ids"], label_info["neg_ids"]
    zA = float(logits[label_info["pos_primary"]])
    zB = float(logits[label_info["neg_primary"]])
    s_lse = float(torch.logsumexp(logits[pos], 0) - torch.logsumexp(logits[neg], 0))
    top1 = int(torch.argmax(logits))
    n_tok = n_audio_tokens(n_samples)
    fam = inputs.get("feature_attention_mask") if hasattr(inputs, "get") else None
    if fam is not None:
        mel = int(fam[0].sum())
        n_tok_mask = ((mel - 1) // 2 + 1 - 2) // 2 + 1
        if n_tok_mask != n_tok:
            raise RuntimeError(f"audio token count mismatch: mask {n_tok_mask} vs samples {n_tok}")
    enc = recorder.data["audio_tower"][0].float()
    proj = recorder.data["projector"][0].float()
    if enc.shape[0] < n_tok:
        raise RuntimeError(f"encoder returned {enc.shape[0]} positions, expected at least {n_tok}")
    enc_v, proj_v = enc[:n_tok], proj[:n_tok]
    idx = torch.as_tensor(region_token_indices(segment_ms, n_tok, n_samples / sr * 1000.0),
                          device=enc_v.device)
    feats = {
        "h_final": recorder.data["final_norm"][0].float(),
        "enc_global_mean": enc_v.mean(0),
        "enc_region_mean": enc_v.index_select(0, idx).mean(0),
        "enc_region_max": enc_v.index_select(0, idx).max(0).values,
        "proj_region_mean": proj_v.index_select(0, idx).mean(0),
    }
    for k, v in recorder.data.items():
        if k.startswith("layer"):
            feats[f"h_{k}"] = v[0].float()
    scal = {"zA": zA, "zB": zB, "s_primary": zA - zB, "s_lse": s_lse, "top1": top1,
            "top1_is_label": bool(top1 in pos or top1 in neg), "n_audio_tokens": n_tok,
            "region_k0": int(idx[0]), "region_k1": int(idx[-1])}
    if audio_token_id is not None:
        scal["n_audio_in_ids"] = int((inputs["input_ids"][0] == audio_token_id).sum())
    if check_head:
        h = recorder.data["final_norm"][0:1]
        w = recorder.lm_head.weight
        with torch.inference_mode():
            head = recorder.lm_head(h.to(device=w.device, dtype=w.dtype)).float()[0]
        ids = [label_info["pos_primary"], label_info["neg_primary"]]
        scal["head_check_maxdiff"] = float((head[ids].to(logits.device) - logits[ids]).abs().max())
    arrays = {k: v.detach().cpu().numpy().astype(np.float16) for k, v in feats.items()}
    return scal, arrays


# ---------------------------------------------------------------- LoRA loading

_RENAMES_TO_V5 = [  # transformers 4.x module paths inside a PEFT wrapper -> 5.x
    (r"^base_model\.model\.language_model\.model\.", "base_model.model.model.language_model."),
    (r"^base_model\.model\.language_model\.lm_head", "base_model.model.lm_head"),
    (r"^base_model\.model\.audio_tower\.", "base_model.model.model.audio_tower."),
    (r"^base_model\.model\.multi_modal_projector\.", "base_model.model.model.multi_modal_projector."),
]
_RENAMES_TO_V4 = [
    (r"^base_model\.model\.model\.language_model\.", "base_model.model.language_model.model."),
    (r"^base_model\.model\.lm_head", "base_model.model.language_model.lm_head"),
    (r"^base_model\.model\.model\.audio_tower\.", "base_model.model.audio_tower."),
    (r"^base_model\.model\.model\.multi_modal_projector\.", "base_model.model.multi_modal_projector."),
]


def read_adapter_state(adapter_dir) -> dict:
    p = Path(adapter_dir)
    if (p / "adapter_model.safetensors").exists():
        from safetensors.torch import load_file
        return load_file(str(p / "adapter_model.safetensors"))
    import torch
    return torch.load(str(p / "adapter_model.bin"), map_location="cpu")


def _rename(keys_to_tensors: dict, rules) -> dict:
    out = {}
    for k, v in keys_to_tensors.items():
        nk = k
        for pat, rep in rules:
            nk = re.sub(pat, rep, nk)
        out[nk] = v
    return out


def lora_report(model) -> dict:
    """LoRA modules per component and how many still hold the zero init of B."""
    total, zero, comps = 0, 0, {}
    for name, mod in model.named_modules():
        lb = getattr(mod, "lora_B", None)
        if lb is None or not hasattr(lb, "values") or len(lb) == 0:
            continue
        total += 1
        comp = ("audio_tower" if "audio_tower" in name else
                "language_model" if "language_model" in name else "other")
        comps[comp] = comps.get(comp, 0) + 1
        if all(float(w.weight.detach().abs().sum()) == 0.0 for w in lb.values()):
            zero += 1
    return {"n_lora_modules": total, "by_component": comps, "n_lora_B_zero": zero}


def load_lora(model, adapter_dir, adapter_name: str = "default"):
    """Attach a trained LoRA adapter and verify that its weights really loaded.

    Adapters saved with transformers 4.x use module paths that differ from the
    5.x layout. If PEFT leaves LoRA B matrices at their zero init, the state dict
    is reloaded under the alternative naming and checked again.
    """
    from peft import PeftModel, set_peft_model_state_dict
    pm = PeftModel.from_pretrained(model, str(adapter_dir), adapter_name=adapter_name, is_trainable=False)
    rep = lora_report(pm)
    rep["remapped"] = None
    if rep["n_lora_modules"] == 0:
        raise RuntimeError("no LoRA modules were created; check target_modules in adapter_config.json")
    if rep["n_lora_B_zero"] > 0:
        sd = read_adapter_state(adapter_dir)
        for tag, rules in (("to_v5", _RENAMES_TO_V5), ("to_v4", _RENAMES_TO_V4)):
            set_peft_model_state_dict(pm, _rename(sd, rules), adapter_name=adapter_name)
            rep = lora_report(pm)
            rep["remapped"] = tag
            if rep["n_lora_B_zero"] == 0:
                break
    if rep["n_lora_B_zero"] > 0:
        raise RuntimeError(f"adapter weights did not load into {rep['n_lora_B_zero']} LoRA modules: {rep}")
    pm.eval()
    return pm, rep


# ---------------------------------------------------------------- feature store

class FeatureStore:
    """Append-only store: per-item scalars as CSV shards, vectors as npz shards.

    A shard is committed when its CSV exists; the npz is written first. On
    restart, items whose variant_id appears in a committed CSV are skipped.
    """

    def __init__(self, out_dir, shard_size: int = 250):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self._meta, self._arr = [], {}

    def _indices(self):
        return sorted(int(p.stem.split("_")[1]) for p in self.dir.glob("shard_*.csv"))

    def done_ids(self) -> set:
        ids = set()
        for k in self._indices():
            ids.update(pd.read_csv(self.dir / f"shard_{k:05d}.csv", usecols=["variant_id"])
                       ["variant_id"].astype(str))
        return ids

    def add(self, meta: dict, arrays: dict):
        self._meta.append(meta)
        for k, v in arrays.items():
            self._arr.setdefault(k, []).append(v)
        if len(self._meta) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self._meta:
            return
        existing = [int(p.stem.split("_")[1]) for p in self.dir.glob("shard_*")]
        k = max(existing) + 1 if existing else 0
        stem = self.dir / f"shard_{k:05d}"
        tmp_npz, tmp_csv = self.dir / f".tmp_{k:05d}.npz", self.dir / f".tmp_{k:05d}.csv"
        with open(tmp_npz, "wb") as fh:
            np.savez(fh, **{n: np.stack(v) for n, v in self._arr.items()})
        os.replace(tmp_npz, stem.with_suffix(".npz"))
        pd.DataFrame(self._meta).to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, stem.with_suffix(".csv"))
        self._meta, self._arr = [], {}


def load_store(out_dir):
    """All committed shards as (meta DataFrame, dict of stacked float16 arrays)."""
    d = Path(out_dir)
    metas, arrays = [], {}
    for p in sorted(d.glob("shard_*.csv")):
        m = pd.read_csv(p)
        npz = np.load(p.with_suffix(".npz"))
        for k in npz.files:
            if len(npz[k]) != len(m):
                raise RuntimeError(f"{p.name}: {k} has {len(npz[k])} rows, meta has {len(m)}")
            arrays.setdefault(k, []).append(npz[k])
        metas.append(m)
    if not metas:
        raise FileNotFoundError(f"no committed shards in {d}")
    meta = pd.concat(metas, ignore_index=True)
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    if meta["variant_id"].duplicated().any():
        keep = ~meta["variant_id"].duplicated(keep="last").to_numpy()
        meta = meta[keep].reset_index(drop=True)
        arrays = {k: v[keep] for k, v in arrays.items()}
    return meta, arrays


def run_extraction(model, processor, recorder: Recorder, manifest: pd.DataFrame, store: FeatureStore,
                   prompt_key: str, label_info: dict, audio_token_id: Optional[int] = None,
                   max_items: Optional[int] = None, n_head_checks: int = 3, log_every: int = 100):
    """Extract features for every manifest row not yet in the store."""
    done = store.done_ids()
    todo = manifest[~manifest["variant_id"].astype(str).isin(done)]
    if max_items is not None:
        todo = todo.head(max_items)
    text = build_text(processor, PROMPTS[prompt_key])
    dev = model_device(model)
    t0, n = time.time(), len(todo)
    print(f"{len(done)} items already stored, {n} to extract")
    for i, row in enumerate(todo.itertuples(index=False)):
        wav = load_wav(row.path)
        inputs = prepare_inputs(processor, text, wav).to(dev)
        scal, arrays = forward_collect(model, recorder, inputs, label_info, segment_ms=row.duration_ms,
                                       n_samples=len(wav), audio_token_id=audio_token_id,
                                       check_head=i < n_head_checks)
        meta = {c: getattr(row, c) for c in META_COLS}
        meta.update(scal)
        meta["prompt"] = prompt_key
        store.add(meta, arrays)
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{n}  {el / (i + 1):.3f} s/item  ETA {(n - i - 1) * el / (i + 1) / 60:.1f} min")
    store.flush()
    print(f"done in {(time.time() - t0) / 60:.1f} min")


# ============================================================================
# D. Readouts
# ============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


L1_METHODS = ("none", "bc_global", "bc_cond", "center_cond", "median_cond", "cf")
BATCH_RULES = ("bc_global", "bc_cond", "center_cond", "median_cond")


def weighted_median(x, w=None) -> float:
    """Median of x with non-negative weights. With unit weights it equals
    np.median; with integer weights it equals the median of the expanded sample."""
    x = np.asarray(x, dtype=float)
    w = np.ones(len(x)) if w is None else np.asarray(w, dtype=float)
    keep = w > 0
    x, w = x[keep], w[keep]
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    cum = np.cumsum(w)
    half = cum[-1] / 2.0
    k = int(np.searchsorted(cum, half - 1e-9))
    if abs(cum[k] - half) < 1e-9 and k + 1 < len(x):
        return float(0.5 * (x[k] + x[k + 1]))
    return float(x[k])


def _rule_value(s, w, rule: str) -> float:
    if rule in ("bc_cond", "bc_global"):
        return float(logit(np.sum(w * sigmoid(s)) / np.sum(w)))
    if rule == "center_cond":
        return float(np.sum(w * s) / np.sum(w))
    if rule == "median_cond":
        return weighted_median(s, w)
    raise ValueError(f"unknown rule {rule}")


def apply_rule(meta: pd.DataFrame, s, rule: str, w=None, group_col: str = "condition") -> np.ndarray:
    """Thresholds of a batch rule estimated from the (weighted) unlabeled batch.

    bc_global is estimated once over all items; the *_cond rules once per
    condition. Called with bootstrap weights, it re-estimates the offset in
    every replicate, so its uncertainty enters the intervals.
    """
    s = np.asarray(s, dtype=float)
    w = np.ones(len(s)) if w is None else np.asarray(w, dtype=float)
    thr = np.full(len(s), np.nan)
    if rule == "bc_global":
        thr[:] = _rule_value(s, w, rule)
        return thr
    groups = meta[group_col].astype(str).to_numpy()
    for g in pd.unique(groups):
        m = groups == g
        if w[m].sum() > 0:
            thr[m] = _rule_value(s[m], w[m], rule)
    return thr


def l1_thresholds(meta: pd.DataFrame, s, method: str, group_col: str = "condition", s_cf=None) -> np.ndarray:
    """Label-free decision thresholds on the native score s = zA - zB.

    none         L0, the native rule s > 0
    bc_global    Batch Calibration over the whole evaluation set
    bc_cond      Batch Calibration per condition (transductive, needs unlabeled
                 batches per condition and a balanced prior)
    center_cond  per-condition mean of s (logit centering)
    median_cond  per-condition median of s (prior matching at 50/50)
    cf           mean score on speech-free containers (Zhao et al. 2021 style)

    With the two label probabilities renormalised, p(A|x) = sigmoid(s). Batch
    Calibration's subtractive and ratio forms then coincide: predict A iff
    p(A|x) > mean_batch p(A|x), i.e. s > logit(mean sigmoid(s)). Every variant
    is a threshold on the same score, so AUROC is unchanged by construction.
    """
    s = np.asarray(s, dtype=float)
    if method == "none":
        return np.zeros_like(s)
    if method == "cf":
        if s_cf is None or len(s_cf) == 0:
            raise ValueError("method 'cf' needs content-free scores")
        return np.full_like(s, float(np.mean(s_cf)))
    if method in BATCH_RULES:
        return apply_rule(meta, s, method, group_col=group_col)
    raise ValueError(f"unknown method {method}")


def readout_parts(r):
    """(score, threshold, rule) from a 2- or 3-tuple readout."""
    return (r[0], r[1], r[2] if len(r) > 2 else None)


def _head(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def fit_head(X, y, groups, Cs=(1e-4, 1e-3, 1e-2, 1e-1), n_splits: int = 4, seed: int = 0):
    """L2-regularised logistic head; C chosen by grouped CV on AUROC."""
    from sklearn.model_selection import StratifiedGroupKFold
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups[y == 1])), len(np.unique(groups[y == 0])))
    best = None
    if n_splits >= 2 and len(Cs) > 1:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for C in Cs:
            aucs = [auc_w(y[va], _head(C).fit(X[tr], y[tr]).decision_function(X[va]))
                    for tr, va in cv.split(X, y, groups)]
            score = float(np.nanmean(aucs))
            if best is None or score > best[1] + 1e-12:
                best = (C, score)
    else:
        best = (Cs[len(Cs) // 2], float("nan"))
    model = _head(best[0]).fit(X, y)
    return model, {"C": best[0], "cv_auroc": best[1]}


def crossfit_scores(X, y, groups, n_splits: int = 5, seed: int = 0, Cs=(1e-4, 1e-3, 1e-2, 1e-1),
                    inner_splits: int = 4):
    """Out-of-fold head scores; folds never split a source recording."""
    from sklearn.model_selection import StratifiedGroupKFold
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    scores = np.full(len(y), np.nan)
    infos = []
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(cv.split(X, y, groups)):
        if set(groups[tr]) & set(groups[te]):
            raise RuntimeError("recording leakage between folds")
        model, info = fit_head(X[tr], y[tr], groups[tr], Cs=Cs, n_splits=inner_splits, seed=seed + f + 1)
        scores[te] = model.decision_function(X[te])
        infos.append(info)
    return scores, infos


def build_readouts(meta: pd.DataFrame, arrays: dict, s_key: str = "s_primary", s_cf=None,
                   head_feature: str = "h_final",
                   probe_features: Sequence[str] = ("enc_region_mean", "enc_global_mean"),
                   train: Optional[tuple] = None, n_splits: int = 5, seed: int = 0,
                   Cs=(1e-4, 1e-3, 1e-2, 1e-1)):
    """Readouts as {name: (score, threshold, rule)} plus fitting info.

    rule names a batch rule that the bootstrap re-estimates per replicate;
    None marks a fixed threshold.
    train=None   heads are cross-fitted inside the evaluation set. They use
                 evaluation labels the LoRA never saw, so they are named
                 L2_cv and P_<feature>_cv and are exploratory only.
    train=(meta_tr, arrays_tr)  heads are fitted on separate supervision, e.g.
                 the variants the LoRA was trained on. Named L2 and
                 P_<feature>; this is the confirmatory, matched comparison.
    """
    y = meta["label"].to_numpy(int)
    s = meta[s_key].to_numpy(float)
    zero = np.zeros(len(meta))
    R = {"L0": (s, zero, None)}
    for m in BATCH_RULES:
        R[f"L1_{m}"] = (s, l1_thresholds(meta, s, m), m)
    if s_cf is not None and len(s_cf):
        R["L1_cf"] = (s, l1_thresholds(meta, s, "cf", s_cf=s_cf), None)
    info = {}
    for feat in (head_feature,) + tuple(probe_features):
        if feat not in arrays:
            continue
        X = arrays[feat].astype(np.float32)
        if train is None:
            sc, inf = crossfit_scores(X, y, meta["rec_id"].to_numpy(), n_splits=n_splits, seed=seed, Cs=Cs)
        else:
            mtr, atr = train
            model, inf = fit_head(atr[feat].astype(np.float32), mtr["label"].to_numpy(int),
                                  mtr["rec_id"].to_numpy(), Cs=Cs, seed=seed)
            sc = model.decision_function(X)
        name = "L2" if feat == head_feature else f"P_{feat}"
        if train is None:
            name += "_cv"
        R[name] = (sc, zero.copy(), None)
        info[name] = inf
    return R, info


def head_name(readouts: dict, base: str) -> str:
    """The confirmatory name if present, else its cross-fitted variant."""
    if base in readouts:
        return base
    if base + "_cv" in readouts:
        return base + "_cv"
    raise KeyError(f"neither {base} nor {base}_cv in readouts")


def combine_runs(runs: dict):
    """Align several runs on their shared items for one paired bootstrap.

    runs: {tag: (meta, readouts)}, e.g. {"base": ..., "lora": ...}. Returns
    (meta, readouts) where readouts are renamed '<tag>:<name>' and every array
    follows the row order of the returned meta. Items must agree on label,
    condition and recording across runs.
    """
    tags = list(runs)
    first = runs[tags[0]][0]
    common = set(first["variant_id"].astype(str))
    for t in tags[1:]:
        common &= set(runs[t][0]["variant_id"].astype(str))
    meta = first[first["variant_id"].astype(str).isin(common)].reset_index(drop=True)
    order = meta["variant_id"].astype(str).to_numpy()
    out = {}
    for t, (m, R) in runs.items():
        pos = pd.Series(np.arange(len(m)), index=m["variant_id"].astype(str).to_numpy()).loc[order].to_numpy()
        for col in ("label", "condition", "rec_id"):
            if not (m[col].to_numpy()[pos] == meta[col].to_numpy()).all():
                raise ValueError(f"run {t} disagrees on {col}")
        for name, r in R.items():
            s, thr, rule = readout_parts(r)
            out[f"{t}:{name}"] = (np.asarray(s)[pos], np.asarray(thr)[pos], rule)
    return meta, out


def did_contrast(conditions: Sequence[str], a_plus: str, a_minus: str, b_plus: str, b_minus: str,
                 metric: str = "ba") -> list:
    """Terms for the mean over conditions of (a_plus - a_minus) - (b_plus - b_minus)."""
    k = 1.0 / len(conditions)
    return [t for c in conditions for t in ((k, a_plus, c, metric), (-k, a_minus, c, metric),
                                            (-k, b_plus, c, metric), (k, b_minus, c, metric))]


def prevalence_check(meta: pd.DataFrame, s, prevalences=(0.2, 0.5, 0.8), reps: int = 50, seed: int = 0,
                     rules: Sequence[str] = ("bc_cond", "median_cond")) -> pd.DataFrame:
    """BA when each condition's unlabeled batch has a given speech prevalence.

    Within every condition, replicates subsample one class to reach the target
    prevalence and re-estimate the batch rules on that batch. A fixed threshold
    (L0) keeps its expected BA; batch-estimated thresholds move with the
    prevalence. Secondary check of the 50/50 assumption of bc_cond.
    """
    rng = np.random.default_rng(seed)
    y = meta["label"].to_numpy(int)
    s = np.asarray(s, dtype=float)
    rows = []
    for (axis, cond), idx in meta.groupby(["axis", "condition"], sort=False).indices.items():
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        for p in prevalences:
            n_pos = len(pos)
            n_neg = int(round(n_pos * (1 - p) / p))
            if n_neg > len(neg):
                n_neg = len(neg)
                n_pos = int(round(n_neg * p / (1 - p)))
            for r in range(reps):
                sel = np.r_[rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
                yy, ss = y[sel], s[sel]
                row = {"axis": axis, "condition": cond, "prevalence": p,
                       "L0": ba_w(yy, ss > 0), "oracle": oracle_ba_w(yy, ss)}
                for rule in rules:
                    row[f"L1_{rule}"] = ba_w(yy, ss > _rule_value(ss, np.ones(len(ss)), rule))
                rows.append(row)
    df = pd.DataFrame(rows)
    return df.groupby(["axis", "condition", "prevalence"], sort=False).mean(numeric_only=True).reset_index()


# ============================================================================
# E. Metrics
# ============================================================================

def auc_w(y, s, w=None) -> float:
    """Weighted AUROC (Mann-Whitney with ties counted half)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    w = np.ones(len(s)) if w is None else np.asarray(w, dtype=float)
    wp, wn = w * y, w * ~y
    Wp, Wn = wp.sum(), wn.sum()
    if Wp <= 0 or Wn <= 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    start = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    wp_g = np.add.reduceat(wp[order], start)
    wn_g = np.add.reduceat(wn[order], start)
    below = np.cumsum(wn_g) - wn_g
    return float(np.sum(wp_g * (below + 0.5 * wn_g)) / (Wp * Wn))


def ba_w(y, pred, w=None) -> float:
    y = np.asarray(y).astype(bool)
    p = np.asarray(pred).astype(bool)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    Wp, Wn = (w * y).sum(), (w * ~y).sum()
    if Wp <= 0 or Wn <= 0:
        return float("nan")
    return float(0.5 * ((w * (y & p)).sum() / Wp + (w * (~y & ~p)).sum() / Wn))


def oracle_ba_w(y, s, w=None) -> float:
    """Best BA over all thresholds on s (L1*). Optimistic by design."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    w = np.ones(len(s)) if w is None else np.asarray(w, dtype=float)
    Wp, Wn = (w * y).sum(), (w * ~y).sum()
    if Wp <= 0 or Wn <= 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ss = s[order]
    end = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    tpr = np.cumsum((w * y)[order])[end] / Wp
    fpr = np.cumsum((w * ~y)[order])[end] / Wn
    return float(max(0.5, np.max(0.5 * (tpr + 1.0 - fpr))))


def d_a_from_auc(a, eps: float = 1e-4):
    from scipy.stats import norm
    a = np.clip(np.asarray(a, dtype=float), eps, 1 - eps)
    return np.sqrt(2.0) * norm.ppf(a)


def rel_efficiency(da_x, da_ref):
    """Relative decoding efficiency (d_a,x / d_a,ref)^2, relative to a reference probe.

    Not Tanner and Birdsall's absolute efficiency: the reference is not an
    ideal observer, so values above 1 are possible and informative.
    """
    da_x = np.asarray(da_x, dtype=float)
    da_ref = np.asarray(da_ref, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(da_ref > 0, (da_x / da_ref) ** 2, np.nan)


def sdt_counts(tp, fn, fp, tn) -> dict:
    """Equal-variance SDT with the log-linear correction (+0.5, +1)."""
    from scipy.stats import norm
    H = (tp + 0.5) / (tp + fn + 1.0)
    F = (fp + 0.5) / (fp + tn + 1.0)
    zH, zF = norm.ppf(H), norm.ppf(F)
    return {"H": H, "F": F, "dprime": zH - zF, "c": -0.5 * (zH + zF)}


def sdt_from_pred(y, pred) -> dict:
    y = np.asarray(y).astype(bool)
    p = np.asarray(pred).astype(bool)
    return sdt_counts(int((y & p).sum()), int((y & ~p).sum()), int((~y & p).sum()), int((~y & ~p).sum()))


def _level(axis: str, row) -> float:
    if axis == "duration":
        return float(row["duration_ms"])
    if axis == "snr":
        return float(row["snr_db"])
    if axis == "reverb":
        return float(row["t60_s"])
    return float("nan")


def condition_table(meta: pd.DataFrame, readouts: dict) -> pd.DataFrame:
    """One row per (axis, condition, readout): AUROC, d_a, oracle BA, BA, SDT."""
    y = meta["label"].to_numpy(int)
    rows = []
    for (axis, cond), idx in meta.groupby(["axis", "condition"], sort=False).indices.items():
        first = meta.iloc[idx[0]]
        lvl = _level(axis, first)
        for name, r in readouts.items():
            s, thr, _ = readout_parts(r)
            ss, tt, yy = s[idx], thr[idx], y[idx]
            pred = ss > tt
            a = auc_w(yy, ss)
            r = {"axis": axis, "condition": cond, "level": lvl, "readout": name, "n": len(idx),
                 "auroc": a, "d_a": float(d_a_from_auc(a)) if not np.isnan(a) else np.nan,
                 "oracle_ba": oracle_ba_w(yy, ss), "ba": ba_w(yy, pred)}
            r.update(sdt_from_pred(yy, pred))
            rows.append(r)
    return pd.DataFrame(rows)


def pivot(table: pd.DataFrame, metric: str, axis: str) -> pd.DataFrame:
    """Conditions of one axis (ordered by level) by readouts, for quick reading."""
    sub = table[table["axis"] == axis]
    p = sub.pivot_table(index=["level", "condition"], columns="readout", values=metric, sort=False)
    return p.sort_index(level=0).round(3)


def mean_contrast(conditions: Sequence[str], plus: str, minus: str, metric: str) -> list:
    """Terms for the mean over conditions of metric(plus) - metric(minus)."""
    k = 1.0 / len(conditions)
    return [t for c in conditions for t in ((k, plus, c, metric), (-k, minus, c, metric))]


def efficiency_table(table: pd.DataFrame, readout: str = "L0", reference: str = "P_enc_region_mean") -> pd.DataFrame:
    a = table[table["readout"] == readout].set_index(["axis", "condition"])
    b = table[table["readout"] == reference].set_index(["axis", "condition"])
    j = a[["level", "d_a"]].join(b[["d_a"]], rsuffix="_ref", how="inner")
    j["rel_efficiency"] = rel_efficiency(j["d_a"], j["d_a_ref"])
    j["readout"], j["reference"] = readout, reference
    return j.reset_index()


# ============================================================================
# F. Bootstrap
# ============================================================================

def cluster_bootstrap(meta: pd.DataFrame, readouts: dict, B: int = 2000, seed: int = 0,
                      metrics: Sequence[str] = ("auroc", "ba"), cluster_col: str = "rec_id",
                      strata_col: str = "label", contrasts: Optional[dict] = None, alpha: float = 0.05):
    """Paired cluster bootstrap over source recordings, stratified by class.

    Every replicate resamples recordings with replacement within each class and
    reweights all their items, across all conditions and readouts at once, so
    differences between readouts, conditions or (after combine_runs) backbones
    are paired. Readouts that carry a batch rule have their thresholds
    re-estimated from each replicate's resampled, unlabeled batch.

    contrasts: {name: [(coef, readout, condition, metric), ...]}, a linear
    combination evaluated on every replicate.

    Returns (ci, contrast_ci). contrast_ci holds the estimate, percentile CI,
    the share of replicates <= 0, the bootstrap SE and mde80 = 2.80 * SE, the
    smallest true effect a two-sided 5% test would detect with 80% power.
    """
    y = meta["label"].to_numpy(int)
    clusters = meta[cluster_col].astype(str).to_numpy()
    uniq, inv = np.unique(clusters, return_inverse=True)
    strat = meta.groupby(cluster_col)[strata_col].first().reindex(uniq).to_numpy()
    groups = list(meta.groupby(["axis", "condition"], sort=False).indices.items())
    keys = [(axis, cond, name, met) for (axis, cond), _ in groups for name in readouts for met in metrics]
    col = {k: i for i, k in enumerate(keys)}
    parts = {name: readout_parts(r) for name, r in readouts.items()}

    def stats(w):
        out = np.empty(len(keys))
        thrs = {name: (thr if rule is None else apply_rule(meta, s, rule, w))
                for name, (s, thr, rule) in parts.items()}
        for (axis, cond), idx in groups:
            yy, ww = y[idx], (None if w is None else w[idx])
            for name, (s, _, _) in parts.items():
                if "auroc" in metrics:
                    out[col[(axis, cond, name, "auroc")]] = auc_w(yy, s[idx], ww)
                if "ba" in metrics:
                    out[col[(axis, cond, name, "ba")]] = ba_w(yy, s[idx] > thrs[name][idx], ww)
                if "oracle_ba" in metrics:
                    out[col[(axis, cond, name, "oracle_ba")]] = oracle_ba_w(yy, s[idx], ww)
        return out

    point = stats(None)
    rng = np.random.default_rng(seed)
    strata_ids = [np.flatnonzero(strat == st) for st in np.unique(strat)]
    reps = np.empty((B, len(keys)))
    for b in range(B):
        counts = np.zeros(len(uniq))
        for ids in strata_ids:
            np.add.at(counts, rng.choice(ids, size=len(ids), replace=True), 1.0)
        reps[b] = stats(counts[inv])
    lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    ci = pd.DataFrame([{"axis": k[0], "condition": k[1], "readout": k[2], "metric": k[3],
                        "estimate": point[i], "lo": lo[i], "hi": hi[i]} for k, i in col.items()])
    rows = []
    for name, terms in (contrasts or {}).items():
        cond_axis = {c: a for (a, c), _ in groups}
        vec_p, vec_r = 0.0, np.zeros(B)
        for coef, rd, cond, met in terms:
            i = col[(cond_axis[cond], cond, rd, met)]
            vec_p = vec_p + coef * point[i]
            vec_r = vec_r + coef * reps[:, i]
        l, h = np.nanpercentile(vec_r, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        se = float(np.nanstd(vec_r, ddof=1))
        rows.append({"contrast": name, "estimate": vec_p, "lo": l, "hi": h,
                     "p_le_0": float(np.mean(vec_r <= 0)), "se": se, "mde80": 2.8016 * se})
    return ci, pd.DataFrame(rows)


# ============================================================================
# G. Thresholds
# ============================================================================

def crossing(levels, values, criterion: float, transform: str = "log2") -> dict:
    """Sustained crossing on an axis where larger levels are easier.

    status  ok              one crossing, interpolated between the last level
                            below and the next level
            ambiguous       several crossings; the sustained crossing is still
                            reported when the easiest level meets the criterion
            censored_below  criterion met at every level (threshold <= min level)
            censored_above  criterion never met (threshold > max level)
    transform: 'log2' for duration (interpolate in log2 ms), 'linear' for dB.
    """
    x = np.asarray(levels, dtype=float)
    v = np.asarray(values, dtype=float)
    order = np.argsort(x)
    x, v = x[order], v[order]
    above = v >= criterion
    n_cross = int(np.sum(above[1:] != above[:-1]))
    if above.all():
        return {"threshold": float(x[0]), "status": "censored_below", "n_crossings": 0}
    if not above.any():
        return {"threshold": float("nan"), "status": "censored_above", "n_crossings": 0}
    if not above[-1]:
        return {"threshold": float("nan"), "status": "ambiguous", "n_crossings": n_cross}
    k = int(np.flatnonzero(~above).max())
    fwd = np.log2 if transform == "log2" else (lambda z: z)
    inv = (lambda z: 2.0 ** z) if transform == "log2" else (lambda z: z)
    t = (criterion - v[k]) / (v[k + 1] - v[k])
    thr = float(inv(fwd(x[k]) + t * (fwd(x[k + 1]) - fwd(x[k]))))
    return {"threshold": thr, "status": "ok" if n_cross == 1 else "ambiguous", "n_crossings": n_cross}


def threshold_table(table: pd.DataFrame, metric: str, criterion: float, axis: str) -> pd.DataFrame:
    sub = table[table["axis"] == axis]
    rows = []
    for name, g in sub.groupby("readout", sort=False):
        r = crossing(g["level"], g[metric], criterion, "log2" if axis == "duration" else "linear")
        r.update({"readout": name, "axis": axis, "metric": metric, "criterion": criterion})
        rows.append(r)
    return pd.DataFrame(rows)
