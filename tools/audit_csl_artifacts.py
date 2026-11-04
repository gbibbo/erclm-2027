#!/usr/bin/env python3
"""Quick audit of the artefacts of the study under review before freezing the ERC-LM protocol.

Checks two things that the design of the workshop study depends on:
  1. Which modules the LoRA adapter actually modified (encoder, projector or LLM).
  2. What the LoRA's train_metadata.csv contains (clean clips or degraded
     variants, how many base clips and how many source recordings).
Optionally it crosses the training set with a test CSV to find shared source
recordings (possible leakage by recording or speaker).

Usage (any argument can be omitted):
  python audit_csl_artifacts.py \
      --adapter results/20260204_201138_COMPARATIVE_RUN/00_lora_training/checkpoints/final \
      --train data/processed/experimental_variants/train_metadata.csv \
      --test data/processed/base_validated_1000/test_base.csv

--test accepts any CSV with a column of audio paths, for example a
predictions.csv of the repository of the study under review.
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ----------------------------------------------------------------------------
# 1. Adapter
# ----------------------------------------------------------------------------

def load_adapter_keys(path: Path):
    if path.is_dir():
        for name in ("adapter_model.safetensors", "adapter_model.bin"):
            if (path / name).exists():
                path = path / name
                break
        else:
            sys.exit(f"Found neither adapter_model.safetensors nor adapter_model.bin in {path}")
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        # framework "np": only the keys are listed, torch is not needed
        with safe_open(str(path), "np") as f:
            return list(f.keys())
    import torch
    return list(torch.load(str(path), map_location="cpu").keys())


def component_of(key: str) -> str:
    if "audio_tower" in key:
        return "audio_tower (encoder)"
    if "multi_modal_projector" in key:
        return "multi_modal_projector"
    if "language_model" in key:
        return "language_model"
    return "other"


def audit_adapter(path: str):
    print("=" * 70)
    print("1. LoRA ADAPTER")
    print("=" * 70)
    keys = load_adapter_keys(Path(path))
    print(f"Total keys: {len(keys)}")
    by_comp = defaultdict(Counter)
    layers = defaultdict(set)
    for k in keys:
        comp = component_of(k)
        m = re.search(r"\.([A-Za-z0-9_]+)\.lora_[AB]", k)
        by_comp[comp][m.group(1) if m else "?"] += 1
        lm = re.search(r"layers\.(\d+)\.", k)
        if lm:
            layers[comp].add(int(lm.group(1)))
    for comp, mods in by_comp.items():
        n_layers = len(layers[comp]) if layers[comp] else 0
        print(f"  {comp:24s} {sum(mods.values()):5d} keys, {n_layers} layers, modules {dict(mods)}")
    enc = [k for k in keys if "audio_tower" in k]
    if enc:
        print("\n  VERDICT: the adapter DOES modify the audio encoder. Examples:")
        for k in enc[:3]:
            print(f"    {k}")
    else:
        print("\n  VERDICT: the adapter does NOT touch the audio encoder.")


# ----------------------------------------------------------------------------
# 2. Train manifest
# ----------------------------------------------------------------------------

def recording_id(path: str):
    """Source recording from the file name.

    voxconverse_wewoz_0454_1000ms_dur500ms.wav -> vox:wewoz
    esc50_1-45645-A-31_0152_1000ms_dur20ms.wav -> esc:1-45645 (fold-source)
    """
    name = Path(str(path)).name
    m = re.match(r"voxconverse_([A-Za-z0-9]+)_", name)
    if m:
        return "vox:" + m.group(1)
    m = re.match(r"esc50_(\d+)-(\d+)-[A-Za-z]+-\d+_", name)
    if m:
        return f"esc:{m.group(1)}-{m.group(2)}"
    return None


def path_column(df):
    for c in ("audio_path", "path", "filepath", "file", "wav_path", "audio"):
        if c in df.columns:
            return c
    for c in df.columns:
        if "path" in c.lower() or "file" in c.lower():
            return c
    return None


def audit_train(train_csv: str):
    import pandas as pd
    print("=" * 70)
    print("2. TRAINING MANIFEST")
    print("=" * 70)
    df = pd.read_csv(train_csv)
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    if "ground_truth" in df.columns:
        print(f"Classes: {df['ground_truth'].value_counts().to_dict()}")
    if "clip_id" in df.columns:
        per_clip = df.groupby("clip_id").size()
        print(f"Unique base clips (clip_id): {df['clip_id'].nunique()}")
        print(f"Rows per base clip: min {per_clip.min()}, median {per_clip.median():.0f}, max {per_clip.max()}")
    degraded = False
    for col in ("duration_ms", "snr_db"):
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).sort_index()
            print(f"\n{col}:\n{vc.to_string()}")
            if df[col].nunique(dropna=True) > 1:
                degraded = True
    if {"duration_ms", "snr_db"} <= set(df.columns):
        print("\nDuration x SNR table (rows):")
        print(pd.crosstab(df["duration_ms"], df["snr_db"].fillna("none")).to_string())
    pcol = path_column(df)
    if pcol:
        rec = df[pcol].map(recording_id)
        print(f"\nUnique source recordings: {rec.nunique()} "
              f"(unidentified: {rec.isna().sum()} rows)")
        if "ground_truth" in df.columns:
            print(df.assign(rec=rec).groupby("ground_truth")["rec"].nunique().to_string())
    print("\n  VERDICT:",
          "the manifest CONTAINS degraded variants (several durations or SNRs)."
          if degraded else
          "no variation of duration or SNR in the available columns.")
    return df, pcol


# ----------------------------------------------------------------------------
# 3. Train-test overlap by source recording
# ----------------------------------------------------------------------------

def audit_overlap(train_df, train_pcol, test_csv: str):
    import pandas as pd
    print("=" * 70)
    print("3. TRAIN-TEST OVERLAP BY SOURCE RECORDING")
    print("=" * 70)
    test = pd.read_csv(test_csv)
    tcol = path_column(test)
    if not (train_pcol and tcol):
        print("No path column in both CSVs. Skipping the overlap check.")
        return
    tr = set(train_df[train_pcol].map(recording_id).dropna())
    te = set(test[tcol].map(recording_id).dropna())
    shared = tr & te
    print(f"Train recordings: {len(tr)}, test: {len(te)}, shared: {len(shared)}")
    if shared:
        vox = sorted(s for s in shared if s.startswith("vox:"))
        esc = sorted(s for s in shared if s.startswith("esc:"))
        print(f"  VoxConverse shared: {len(vox)} {vox[:10]}")
        print(f"  ESC-50 shared:      {len(esc)} {esc[:10]}")
        print("  Some recordings (and therefore speakers) are on both sides.")
    if "clip_id" in train_df.columns and "clip_id" in test.columns:
        n = len(set(train_df["clip_id"]) & set(test["clip_id"]))
        print(f"shared clip_id: {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", help="folder of the final checkpoint, or the adapter file")
    ap.add_argument("--train", help="the original train_metadata.csv of the LoRA")
    ap.add_argument("--test", help="test CSV with audio paths")
    args = ap.parse_args()
    if not (args.adapter or args.train):
        ap.error("give at least --adapter or --train")
    if args.adapter:
        audit_adapter(args.adapter)
    if args.train:
        df, pcol = audit_train(args.train)
        if args.test:
            audit_overlap(df, pcol, args.test)


if __name__ == "__main__":
    main()
