"""Bank generator v2: test, train and dev audio banks rebuilt from public VoxConverse and ESC-50.

The protocol is in docs/banks.md. Everything is deterministic: inputs are sorted,
every random draw comes from a seeded generator, and the source file and start of every fragment
are recorded in the metadata. File names follow the scheme of the study under review, so
erclm.lib_v2 parses them unchanged. Three faults of the banks of the study under review are fixed:

    reverb    every level is applied, with simulated rooms whose measured T60 is checked
    SNR       the noise is scaled to the RMS of the fragment, not of the 2 s container
    splits    train, dev and test never share a source recording, and share one normalisation

Layout of a bank folder (packed later as <name>.tar):
    <name>/manifest.csv   audio_path, ground_truth (what erclm.banks.fetch reads)
    <name>/metadata.csv   one row per variant with every parameter
    <name>/audio/*.wav    2 s float32 WAV at 16 kHz
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROTOCOL_VERSION = "banks-v2.0"
SR = 16000
BASE_MS = 1000
CONTAINER_MS = 2000
PAD_SIGMA = 1e-4
PEAK = 0.9
FADE_MS = 1.0
CLIP_GUARD = 0.99
DURATIONS = (20, 40, 60, 80, 100, 200, 500, 1000)
SNRS = (-10, -5, 0, 5, 10, 20)
T60S = (0.3, 1.0, 2.5)
FILTERS = ("bandpass", "lowpass", "highpass")
FILTER_HZ = {"bandpass": (300.0, 3400.0), "lowpass": 3400.0, "highpass": 300.0}
FILTER_ORDER = 4
GRID_MS = 100
VOX_COVERAGE = 0.9
VOX_MIN_RMS = 1e-3
ESC_REL_RMS = 0.25
MIN_GAP_S = 1.0
RT60_TOL = 0.10
ROOMS_PER_T60 = 8

DEFAULT_COUNTS = {"test": {"speech": 485, "nonspeech": 485, "speech_recordings": 210},
                  "train": {"speech": 32, "nonspeech": 32},
                  "dev": {"speech": 36, "nonspeech": 36}}


# ============================================================================ conditions and names

def test_conditions() -> list:
    """The 22 one-factor conditions, in the naming of the study under review."""
    return ([f"dur{d}ms" for d in DURATIONS] + [f"snr{s:+d}dB" for s in SNRS]
            + ["reverbnone"] + [f"reverb{t:.1f}s" for t in T60S]
            + ["filternone"] + [f"filter{f}" for f in FILTERS])


def factorial_conditions() -> list:
    """The 48 duration x SNR conditions of the train and dev banks."""
    return [f"dur{d}ms_snr{s:+d}dB" for d in DURATIONS for s in SNRS]


def variant_name(base_id: str, condition: str) -> str:
    return f"{base_id}_{BASE_MS}ms_{condition}"


def variant_seed(base_id: str, condition: str) -> int:
    return int(hashlib.sha256(f"{PROTOCOL_VERSION}|{base_id}|{condition}".encode()).hexdigest()[:12], 16)


# ============================================================================ signal operations

def n_samples(ms: float) -> int:
    return int(round(ms * SR / 1000.0))


def rms(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def normalize_base(x) -> np.ndarray:
    """Remove DC and scale the peak to PEAK. Same for every split."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    peak = np.abs(x).max()
    return x * (PEAK / peak) if peak > 0 else x


def fade(x, ms: float = FADE_MS) -> np.ndarray:
    """Raised-cosine ramps at both ends, so cuts do not click."""
    x = np.array(x, dtype=np.float64)
    k = min(n_samples(ms), len(x) // 2)
    if k > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(k) + 0.5) / k)
        x[:k] *= ramp
        x[-k:] *= ramp[::-1]
    return x


def center_cut(x, ms: float) -> np.ndarray:
    n = n_samples(ms)
    start = (len(x) - n) // 2
    return np.asarray(x[start:start + n], dtype=np.float64)


def place(segment, rng) -> np.ndarray:
    """Centre the segment in a CONTAINER_MS container of Gaussian padding (sigma PAD_SIGMA)."""
    n = n_samples(CONTAINER_MS)
    out = rng.standard_normal(n) * PAD_SIGMA
    k = len(segment)
    start = (n - k) // 2
    out[start:start + k] += segment
    return out


def add_noise(container, segment_rms: float, snr_db: float, rng) -> tuple:
    """White Gaussian noise over the whole container, scaled to the fragment's RMS."""
    sigma = segment_rms / (10.0 ** (snr_db / 20.0))
    return container + rng.standard_normal(len(container)) * sigma, sigma


def reverberate(segment, rir) -> np.ndarray:
    """Convolve, cut to the segment length and match the dry RMS (the procedure of the study under review)."""
    from scipy.signal import fftconvolve
    wet = fftconvolve(segment, rir, mode="full")[:len(segment)]
    r = rms(wet)
    return wet * (rms(segment) / r) if r > 0 else wet


def band_filter(segment, kind: str) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    sos = butter(FILTER_ORDER, FILTER_HZ[kind], btype=kind, fs=SR, output="sos")
    return sosfiltfilt(sos, segment)


def guard(x) -> tuple:
    """Scale the whole container down if it would clip. Ratios (SNR) are unchanged."""
    peak = float(np.abs(x).max())
    if peak > CLIP_GUARD:
        g = CLIP_GUARD / peak
        return x * g, g
    return x, 1.0


def render(base, condition: str, base_id: str, rirs: Optional[dict] = None) -> tuple:
    """One variant of a normalised 1000 ms base clip. Returns (float32 container, parameters)."""
    rng = np.random.default_rng(variant_seed(base_id, condition))
    params = {"duration_ms": BASE_MS, "snr_db": None, "t60_target_s": 0.0, "t60_measured_s": 0.0,
              "room": None, "filter": "none", "seed": variant_seed(base_id, condition)}
    tokens = condition.split("_")
    dur = next((int(t[3:-2]) for t in tokens if t.startswith("dur")), BASE_MS)
    snr = next((float(t[3:-2]) for t in tokens if t.startswith("snr")), None)
    rev = next((t[6:] for t in tokens if t.startswith("reverb")), "none")
    fil = next((t[6:] for t in tokens if t.startswith("filter")), "none")
    seg = fade(center_cut(base, dur))
    if rev != "none":
        t60 = float(rev[:-1])
        rooms = rirs[t60]
        idx = int(rng.integers(len(rooms)))
        seg = reverberate(seg, rooms[idx]["rir"])
        params.update(t60_target_s=t60, t60_measured_s=rooms[idx]["rt60"], room=f"t60_{t60:.1f}_room{idx}")
    if fil != "none":
        seg = band_filter(seg, fil)
        params["filter"] = fil
    seg_rms = rms(seg)
    x = place(seg, rng)
    if snr is not None:
        x, sigma = add_noise(x, seg_rms, snr, rng)
        params.update(snr_db=snr, noise_sigma=sigma)
    x, gain = guard(x)
    params.update(duration_ms=dur, segment_rms=seg_rms, gain=gain)
    return x.astype(np.float32), params


# ============================================================================ rooms

def make_rooms(seed: int = 42, per_t60: int = ROOMS_PER_T60, t60s=T60S, log=print) -> dict:
    """Simulated shoebox rooms (image-source method, deterministic) whose measured T60 is within
    RT60_TOL of the target. Returns {t60: [{"rir", "rt60", "dims", "src", "mic", "absorption"}]}."""
    import pyroomacoustics as pra
    ranges = {0.3: ((3, 6), (3, 6), (2.5, 3.2)), 1.0: ((6, 12), (5, 10), (3, 4)),
              2.5: ((10, 20), (8, 16), (4, 8))}
    rng = np.random.default_rng([seed, 7])
    out = {}
    for t60 in t60s:
        rooms = []
        while len(rooms) < per_t60:
            dims = [float(rng.uniform(*r)) for r in ranges[t60]]
            src = [float(rng.uniform(1.0, d - 1.0)) for d in dims]
            mic = [float(rng.uniform(1.0, d - 1.0)) for d in dims]
            e_abs, max_order = pra.inverse_sabine(t60, dims)
            e_abs = float(e_abs)
            rt60 = None
            for _ in range(6):  # correct the absorption until the measured T60 matches
                room = pra.ShoeBox(dims, fs=SR, materials=pra.Material(min(max(e_abs, 0.01), 0.99)),
                                   max_order=int(min(max_order, 60)), air_absorption=False)
                room.add_source(src)
                room.add_microphone(mic)
                room.compute_rir()
                rir = np.asarray(room.rir[0][0], dtype=np.float64)
                rt60 = float(room.measure_rt60()[0][0])
                if not math.isfinite(rt60):  # decay too short to measure: make the walls harder
                    e_abs *= 0.8
                    continue
                if abs(rt60 - t60) <= RT60_TOL * t60:
                    break
                e_abs *= rt60 / t60
            if rt60 is None or not math.isfinite(rt60) or abs(rt60 - t60) > RT60_TOL * t60:
                log(f"room for T60 {t60} rejected (measured {rt60}), drawing another")
                continue
            onset = int(np.flatnonzero(np.abs(rir) >= 0.1 * np.abs(rir).max())[0])
            rir = rir[onset:]
            rir = rir / np.sqrt(np.sum(rir ** 2))
            rooms.append({"rir": rir, "rt60": round(rt60, 4), "dims": dims, "src": src, "mic": mic,
                          "absorption": e_abs})
            log(f"room T60 {t60} #{len(rooms) - 1}: measured {rt60:.3f} s, dims {np.round(dims, 2).tolist()}")
        out[t60] = rooms
    return out


def save_rooms(rooms: dict, path) -> Path:
    path = Path(path)
    arrays, meta = {}, []
    for t60, lst in rooms.items():
        for i, r in enumerate(lst):
            key = f"t60_{t60:.1f}_room{i}"
            arrays[key] = r["rir"].astype(np.float32)
            meta.append({"key": key, **{k: v for k, v in r.items() if k != "rir"}, "t60_target": t60})
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def load_rooms(path) -> dict:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    npz = np.load(path)
    out = {}
    for m in meta:
        out.setdefault(m["t60_target"], []).append({**m, "rir": npz[m["key"]].astype(np.float64)})
    return out


# ============================================================================ sources

def read_rttm(path) -> list:
    segs = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 5 and p[0] == "SPEAKER":
            segs.append((float(p[3]), float(p[4])))
    return segs


def speech_mask(segments, total_s: float, frame_ms: float = 10.0) -> np.ndarray:
    n = int(math.ceil(total_s * 1000.0 / frame_ms))
    mask = np.zeros(n, dtype=bool)
    for start, dur in segments:
        a = int(math.floor(start * 1000.0 / frame_ms))
        b = int(math.ceil((start + dur) * 1000.0 / frame_ms))
        mask[max(a, 0):min(b, n)] = True
    return mask


def vox_candidates(audio, segments) -> list:
    """Starts (s) of 1000 ms windows on a GRID_MS grid that are at least VOX_COVERAGE speech."""
    total_s = len(audio) / SR
    mask = speech_mask(segments, total_s)
    frames = BASE_MS // 10
    cands = []
    for k in range(0, len(mask) - frames + 1, GRID_MS // 10):
        if mask[k:k + frames].mean() >= VOX_COVERAGE:
            a = n_samples(k * 10)
            if rms(audio[a:a + n_samples(BASE_MS)]) >= VOX_MIN_RMS:
                cands.append(k * 10 / 1000.0)
    return cands


def esc_candidates(audio) -> list:
    """Starts (s) of 1000 ms windows whose RMS is at least ESC_REL_RMS of the loudest window."""
    w, step = n_samples(BASE_MS), n_samples(GRID_MS)
    starts = list(range(0, len(audio) - w + 1, step))
    vals = [rms(audio[s:s + w]) for s in starts]
    top = max(vals) if vals else 0.0
    return [s / SR for s, v in zip(starts, vals) if top > 0 and v >= ESC_REL_RMS * top]


def pick_windows(cands, n: int, rng, min_gap_s: float = MIN_GAP_S) -> list:
    """n window starts drawn at random, no two closer than BASE_MS + min_gap_s."""
    pool = list(cands)
    chosen = []
    while pool and len(chosen) < n:
        s = pool.pop(int(rng.integers(len(pool))))
        chosen.append(s)
        pool = [c for c in pool if abs(c - s) >= BASE_MS / 1000.0 + min_gap_s]
    return sorted(chosen) if len(chosen) == n else []


def load_audio(path, start_s: float = 0.0, dur_s: Optional[float] = None) -> np.ndarray:
    """Mono 16 kHz float64 from any WAV. Resampling uses a fixed polyphase filter."""
    import soundfile as sf
    from fractions import Fraction
    from scipy.signal import resample_poly
    info = sf.info(str(path))
    a = int(round(start_s * info.samplerate))
    frames = -1 if dur_s is None else int(round(dur_s * info.samplerate))
    x, sr = sf.read(str(path), start=a, frames=frames, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        f = Fraction(SR, sr).limit_denominator(1000)
        x = resample_poly(x, f.numerator, f.denominator)
    return x


# ============================================================================ selection

def select_clips(vox: dict, esc: pd.DataFrame, seed: int = 42, counts: dict = None, log=print) -> pd.DataFrame:
    """Base clips for every split, with disjoint source recordings.

    vox: {recording_id: {"wav": path, "rttm": path}}
    esc: ESC-50 meta with filename, fold, target, src_file, take and a 'path' column.
    """
    counts = counts or DEFAULT_COUNTS
    rng = np.random.default_rng(seed)
    rows = []

    # VoxConverse: recordings in a seeded order, test first, then train, then dev
    recs = sorted(vox)
    order = [recs[i] for i in rng.permutation(len(recs))]
    n_test, n_rec = counts["test"]["speech"], counts["test"]["speech_recordings"]
    plan = ([("test", n_test // n_rec + (1 if i < n_test - (n_test // n_rec) * n_rec else 0)) for i in range(n_rec)]
            + [("train", 1)] * counts["train"]["speech"] + [("dev", 1)] * counts["dev"]["speech"])
    it = iter(order)
    for split, k in plan:
        while True:
            rec = next(it, None)
            if rec is None:
                raise RuntimeError("not enough VoxConverse recordings with speech-dense windows")
            audio = load_audio(vox[rec]["wav"])
            starts = pick_windows(vox_candidates(audio, read_rttm(vox[rec]["rttm"])), k, rng)
            if starts:
                break
            log(f"skipping {rec}: fewer than {k} usable windows")
        for s in starts:
            rows.append({"split": split, "label": "SPEECH", "dataset": "voxconverse", "recording": rec,
                         "source_file": Path(vox[rec]["wav"]).name, "start_s": s})

    # ESC-50: sources (fold, src_file) in a seeded order, one clip per source
    esc = esc.sort_values("filename").reset_index(drop=True)
    sources = sorted({(str(r.fold), str(r.src_file)) for r in esc.itertuples()})
    sorder = [sources[i] for i in rng.permutation(len(sources))]
    need = ([("test", 1)] * counts["test"]["nonspeech"] + [("train", 1)] * counts["train"]["nonspeech"]
            + [("dev", 1)] * counts["dev"]["nonspeech"])
    sit = iter(sorder)
    for split, _ in need:
        while True:
            src = next(sit, None)
            if src is None:
                raise RuntimeError("not enough ESC-50 sources")
            takes = esc[(esc.fold.astype(str) == src[0]) & (esc.src_file.astype(str) == src[1])]
            r = takes.iloc[int(rng.integers(len(takes)))]
            audio = load_audio(r["path"])
            cands = esc_candidates(audio)
            if cands:
                break
        rows.append({"split": split, "label": "NONSPEECH", "dataset": "esc50", "recording": f"{src[0]}-{src[1]}",
                     "source_file": r["filename"], "start_s": cands[int(rng.integers(len(cands)))],
                     "esc_category": r.get("category", "")})

    df = pd.DataFrame(rows)
    df["idx"] = df.groupby("split").cumcount()
    df["base_id"] = [
        f"voxconverse_{r.recording}_{r.idx:04d}" if r.dataset == "voxconverse"
        else f"esc50_{Path(r.source_file).stem}_{r.idx:04d}" for r in df.itertuples()]
    for a, b in (("test", "train"), ("test", "dev"), ("train", "dev")):
        shared = set(df.loc[df.split == a, "recording"]) & set(df.loc[df.split == b, "recording"])
        if shared:
            raise RuntimeError(f"{a} and {b} share recordings: {sorted(shared)[:5]}")
    return df


# ============================================================================ building and packing

def build_bank(name: str, clips: pd.DataFrame, conditions: list, vox: dict, esc_paths: dict, rooms,
               out_dir, log=print) -> dict:
    """Render every clip x condition into out_dir/<name>/. Returns a summary."""
    import soundfile as sf
    root = Path(out_dir) / name
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    meta_rows, man_rows = [], []
    for i, c in enumerate(clips.itertuples()):
        src = vox[c.recording]["wav"] if c.dataset == "voxconverse" else esc_paths[c.source_file]
        base = normalize_base(load_audio(src, c.start_s, BASE_MS / 1000.0)[:n_samples(BASE_MS)])
        if len(base) != n_samples(BASE_MS):
            raise RuntimeError(f"{c.base_id}: source too short at {c.start_s}s")
        for cond in conditions:
            x, p = render(base, cond, c.base_id, rooms)
            fname = variant_name(c.base_id, cond) + ".wav"
            sf.write(str(audio_dir / fname), x, SR, subtype="FLOAT")
            man_rows.append({"audio_path": fname, "ground_truth": c.label})
            meta_rows.append({"audio_path": fname, "ground_truth": c.label, "base_id": c.base_id,
                              "split": c.split, "dataset": c.dataset, "recording": c.recording,
                              "source_file": c.source_file, "start_s": round(float(c.start_s), 3),
                              "condition": cond, **p})
        if log and (i + 1) % 100 == 0:
            log(f"{name}: {i + 1}/{len(clips)} base clips rendered")
    pd.DataFrame(man_rows).to_csv(root / "manifest.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(root / "metadata.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return {"name": name, "variants": len(man_rows), "base_clips": int(len(clips)),
            "conditions": len(conditions)}


def pack(name: str, out_dir) -> tuple:
    """<out_dir>/<name>.tar plus .sha256, with members in sorted order."""
    from .storage import sha256_file
    out_dir = Path(out_dir)
    root = out_dir / name
    tar_path = out_dir / f"{name}.tar"
    tmp = tar_path.with_name(tar_path.name + ".part")
    with tarfile.open(tmp, "w") as tf:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(out_dir)).replace(os.sep, "/"))
    os.replace(tmp, tar_path)
    sha = sha256_file(tar_path)
    tar_path.with_name(tar_path.name + ".sha256").write_text(f"{sha}  {tar_path.name}\n")
    return tar_path, sha


# ============================================================================ downloads (network)

def download(url: str, dest: Path, log=print) -> Path:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_name(dest.name + ".part")
    log(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as fh:
        while True:
            block = r.read(1 << 22)
            if not block:
                break
            fh.write(block)
    os.replace(tmp, dest)
    return dest


def fetch_sources(src_dir, pins: dict, log=print) -> tuple:
    """Download and unpack VoxConverse audio and RTTM and ESC-50 at the pinned versions.
    Returns (vox index, ESC-50 meta with paths, {esc filename: path})."""
    src_dir = Path(src_dir)
    vox_dir, esc_dir = src_dir / "voxconverse", src_dir / "esc50"
    for split in ("dev", "test"):
        z = download(pins["voxconverse_audio"][split], src_dir / f"voxconverse_{split}_wav.zip", log)
        mark = vox_dir / f".{split}.unzipped"
        if not mark.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(vox_dir / split)
            mark.touch()
    rttm_zip = download(f"https://github.com/joonson/voxconverse/archive/{pins['voxconverse_rttm_commit']}.zip",
                        src_dir / "voxconverse_rttm.zip", log)
    rttm_dir = vox_dir / "rttm"
    if not rttm_dir.exists():
        with zipfile.ZipFile(rttm_zip) as zf:
            zf.extractall(rttm_dir)
    esc_zip = download(f"https://github.com/karolpiczak/ESC-50/archive/{pins['esc50_commit']}.zip",
                       src_dir / "esc50.zip", log)
    if not esc_dir.exists():
        with zipfile.ZipFile(esc_zip) as zf:
            zf.extractall(esc_dir)
    wavs = {p.stem: p for p in sorted((vox_dir / "dev").rglob("*.wav")) + sorted((vox_dir / "test").rglob("*.wav"))}
    rttms = {p.stem: p for p in sorted(rttm_dir.rglob("*.rttm"))}
    vox = {k: {"wav": wavs[k], "rttm": rttms[k]} for k in sorted(set(wavs) & set(rttms))}
    meta_csv = next(esc_dir.rglob("meta/esc50.csv"))
    esc = pd.read_csv(meta_csv)
    audio_dir = meta_csv.parent.parent / "audio"
    esc["path"] = [str(audio_dir / f) for f in esc.filename]
    log(f"sources: {len(vox)} VoxConverse recordings, {len(esc)} ESC-50 clips")
    return vox, esc, {f: Path(p) for f, p in zip(esc.filename, esc.path)}
