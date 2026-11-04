"""Per-item score table: everything cluster_bootstrap needs, in a few MB.

One row per evaluated item with its metadata, the native logits and, for every
readout, the score and the fixed threshold estimated on the full batch. A JSON
sidecar names the batch rule of each readout, so the bootstrap can re-estimate
label-free thresholds per replicate. load_scores() rebuilds (meta, readouts)
exactly as build_readouts() returned them.

    from erclm import export, lib_v2 as L
    meta, R = export.load_scores("runs/<run_id>/results/base/scores.csv.gz")
    ci, cc = L.cluster_bootstrap(meta, R, B=2000, contrasts=...)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import lib_v2 as L

SCALARS = ["zA", "zB", "s_primary", "s_lse", "top1", "top1_is_label", "session_id", "host"]
SEP = "__"
FORMAT = "erclm-scores-v1"


def sidecar_path(path) -> Path:
    p = Path(path)
    return p.with_name(p.name.split(".")[0] + ".json")


def score_table(meta: pd.DataFrame, readouts: dict) -> pd.DataFrame:
    cols = [c for c in L.META_COLS + SCALARS if c in meta.columns]
    out = meta[cols].reset_index(drop=True).copy()
    for name, r in readouts.items():
        s, thr, _ = L.readout_parts(r)
        out[f"{name}{SEP}score"] = np.asarray(s, dtype=float)
        out[f"{name}{SEP}thr"] = np.asarray(thr, dtype=float)
    return out


def write_scores(path, meta: pd.DataFrame, readouts: dict, info: dict = None) -> Path:
    """Write <name>.csv.gz plus <name>.json. Returns the table path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    score_table(meta, readouts).to_csv(path, index=False, float_format="%.9g")
    side = {"format": FORMAT, "readouts": {n: {"rule": L.readout_parts(r)[2]} for n, r in readouts.items()},
            "n_items": int(len(meta)), "info": info or {}}
    sidecar_path(path).write_text(json.dumps(side, indent=2, sort_keys=True, default=str))
    return path


def load_scores(path):
    """(meta, readouts) from a table written by write_scores."""
    path = Path(path)
    side = json.loads(sidecar_path(path).read_text())
    if side.get("format") != FORMAT:
        raise ValueError(f"{path}: unknown score table format {side.get('format')}")
    df = pd.read_csv(path)
    meta_cols = [c for c in df.columns if SEP not in c]
    meta = df[meta_cols].copy()
    readouts = {}
    for name, spec in side["readouts"].items():
        readouts[name] = (df[f"{name}{SEP}score"].to_numpy(float), df[f"{name}{SEP}thr"].to_numpy(float),
                          spec["rule"])
    return meta, readouts
