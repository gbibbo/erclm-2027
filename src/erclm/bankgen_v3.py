"""Provenance bank v3 (2026-10-21), decision D2 in docs/exploratory_log.md.

In the v2 banks every speech clip comes from VoxConverse and every non-speech clip from
ESC-50, so corpus and class go together. This bank takes both classes from the same
VoxConverse recordings, none of them used by the v2 banks, so a detector that relies on
corpus or channel cues loses them. bankgen.py stays as it is. This module only selects
the clips and reuses bankgen for loading, rendering, naming and packing.

Per recording, up to per_recording speech windows and as many non-speech windows.
    speech      the v2 rule (bankgen.vox_candidates), 1000 ms with at least 90% RTTM speech
    non-speech  1000 ms with no RTTM speech within NS_MARGIN_S on either side, and an RMS of
                at least NS_REL_RMS times the median speech window of the recording, so it
                is not silence (the analogue of the ESC-50 rule)
A recording enters only with as many windows of each class, so the bank stays balanced
within every recording. Conditions are the 8 durations of the test bank, without noise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import bankgen as G

PROTOCOL = "prov-v3.1"
NS_MARGIN_S = 0.25
NS_REL_RMS = 0.25


def duration_conditions() -> list:
    return [f"dur{d}ms" for d in G.DURATIONS]


def window_rms(audio, starts) -> np.ndarray:
    n = G.n_samples(G.BASE_MS)
    return np.array([G.rms(audio[G.n_samples(s * 1000.0):G.n_samples(s * 1000.0) + n]) for s in starts])


def nonspeech_candidates(audio, segments, min_rms: float, margin_s: float = NS_MARGIN_S) -> list:
    """Starts (s) of 1000 ms windows on the GRID_MS grid with no RTTM speech within margin_s
    and an RMS of at least min_rms."""
    total_s = len(audio) / G.SR
    mask = G.speech_mask(segments, total_s)
    frames, pad = G.BASE_MS // 10, int(round(margin_s * 100))
    cands = []
    for k in range(0, len(mask) - frames + 1, G.GRID_MS // 10):
        if mask[max(k - pad, 0):k + frames + pad].any():
            continue
        s = k * 10 / 1000.0
        if window_rms(audio, [s])[0] >= min_rms:
            cands.append(s)
    return cands


def recording_windows(audio, segments, per_recording: int, rng) -> tuple:
    """(speech starts, non-speech starts), both of length per_recording, or two empty lists."""
    speech = G.vox_candidates(audio, segments)
    if len(speech) < per_recording:
        return [], []
    ref = float(np.median(window_rms(audio, speech)))
    ns = nonspeech_candidates(audio, segments, max(G.VOX_MIN_RMS, NS_REL_RMS * ref))
    sp = G.pick_windows(speech, per_recording, rng)
    nsp = G.pick_windows(ns, per_recording, rng)
    if not sp or not nsp:
        return [], []
    return sp, nsp


def select_provenance_clips(vox: dict, exclude, seed: int = 42, per_recording: int = 1,
                            max_recordings=None, log=print) -> pd.DataFrame:
    """Base clips of both classes from the VoxConverse recordings not in exclude.

    vox: {recording_id: {"wav": path, "rttm": path}}, as bankgen.fetch_sources returns it.
    Clips are named voxconverse_<recording>_<idx>, so lib_v2 assigns both classes of a
    recording the same rec_id and every resampling keeps them together.
    """
    rng = np.random.default_rng(seed)
    recs = sorted(set(vox) - set(exclude))
    order = [recs[i] for i in rng.permutation(len(recs))]
    rows, skipped = [], 0
    for rec in order:
        if max_recordings is not None and len({r["recording"] for r in rows}) >= max_recordings:
            break
        audio = G.load_audio(vox[rec]["wav"])
        sp, nsp = recording_windows(audio, G.read_rttm(vox[rec]["rttm"]), per_recording, rng)
        if not sp:
            skipped += 1
            continue
        src = Path(vox[rec]["wav"]).name
        rows += [{"split": "prov", "label": "SPEECH", "dataset": "voxconverse", "recording": rec,
                  "source_file": src, "start_s": s} for s in sp]
        rows += [{"split": "prov", "label": "NONSPEECH", "dataset": "voxconverse", "recording": rec,
                  "source_file": src, "start_s": s} for s in nsp]
    df = pd.DataFrame(rows, columns=["split", "label", "dataset", "recording", "source_file", "start_s"])
    df["idx"] = np.arange(len(df))
    df["base_id"] = [f"voxconverse_{r.recording}_{r.idx:04d}" for r in df.itertuples()]
    log(f"provenance clips: {df['recording'].nunique()} recordings, {len(df)} base clips, "
        f"{skipped} recordings without both classes")
    return df
