"""Audio banks packed as one tar each, with a .sha256 next to it on Drive.

Layout inside <name>.tar
    <name>/manifest.csv     audio_path (file name only), ground_truth
    <name>/audio/*.wav

make_tar() builds the tar from any CSV plus an audio folder. fetch() copies it
from the remote to local disk, checks the sha256 and extracts it once per
session. validate() checks the counts and WAV formats.
"""
from __future__ import annotations

import hashlib
import os
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import lib_v2 as L
from .storage import sha256_file


def make_tar(name: str, manifest_csv, audio_dir, out_dir, path_col: Optional[str] = None,
             label_col: Optional[str] = None) -> tuple:
    """Pack a bank. Returns (tar path, sha256). Fails if any WAV is missing."""
    raw = pd.read_csv(manifest_csv)
    pcol = path_col or L._find_col(raw, ("audio_path", "path", "filepath", "file", "wav_path"))
    lcol = label_col or L._find_col(raw, ("ground_truth", "label", "y"))
    audio_dir, out_dir = Path(audio_dir), Path(out_dir)
    files = [audio_dir / Path(str(p)).name for p in raw[pcol]]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} WAV files missing, e.g. {missing[:3]}")
    labels = L._to_binary(raw[lcol])
    man = pd.DataFrame({"audio_path": [f.name for f in files],
                        "ground_truth": np.where(labels == 1, "SPEECH", "NONSPEECH")})
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / f"{name}.manifest.csv"
    man.to_csv(man_path, index=False)
    tar_path = out_dir / f"{name}.tar"
    tmp = tar_path.with_name(tar_path.name + ".part")
    with tarfile.open(tmp, "w") as tf:
        tf.add(man_path, arcname=f"{name}/manifest.csv")
        for f in files:
            tf.add(f, arcname=f"{name}/audio/{f.name}")
    os.replace(tmp, tar_path)
    man_path.unlink()
    sha = sha256_file(tar_path)
    tar_path.with_name(tar_path.name + ".sha256").write_text(f"{sha}  {tar_path.name}\n")
    return tar_path, sha


def _safe_extract(tar_path: Path, dest: Path):
    dest = dest.resolve()
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if not (m.isfile() or m.isdir()) or os.path.commonpath([dest, target]) != str(dest):
                raise RuntimeError(f"unsafe tar member {m.name!r} in {tar_path.name}")
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # Python without extraction filters
            tf.extractall(dest)


def remote_sha(remote, tar_rel: str) -> str:
    data = remote.get_bytes(tar_rel + ".sha256")
    if data is None:
        raise FileNotFoundError(f"{tar_rel}.sha256 is missing on the remote")
    return data.decode().split()[0].strip().lower()


def fetch(remote, spec: dict, data_dir, role: str, log=print) -> tuple:
    """Bank from the remote to local disk. Returns (manifest, sha256).

    spec: {'tar': 'data/x.tar', 'manifest': 'x/manifest.csv', 'audio': 'x/audio'}
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    sha = remote_sha(remote, spec["tar"])
    ok = data_dir / f".{Path(spec['tar']).name}.{sha[:16]}.ok"
    if not ok.exists():
        local_tar = data_dir / Path(spec["tar"]).name
        log(f"downloading {spec['tar']}")
        remote.get(spec["tar"], local_tar)
        got = sha256_file(local_tar)
        if got != sha:
            raise RuntimeError(f"{spec['tar']}: sha256 {got} does not match {sha}")
        log(f"extracting {local_tar.name}")
        _safe_extract(local_tar, data_dir)
        local_tar.unlink()
        ok.touch()
    man = L.load_manifest(data_dir / spec["manifest"], audio_root=data_dir / spec["audio"], role=role)
    return man, sha


def ids_hash(variant_ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, variant_ids))).encode()).hexdigest()


def validate(man: pd.DataFrame, expect: Optional[dict] = None, check_audio: bool = True,
             names_csv=None, sr: int = L.SR, duration_s: float = L.CONTAINER_MS / 1000.0) -> dict:
    """Counts, balance and WAV format of a manifest. report['failures'] lists what broke.

    expect may hold rows, conditions, base_ids, rec_ids (exact counts) and
    speech_share (0.5 for the balanced test bank).
    names_csv: a CSV with an audio_path column (e.g. a predictions.csv of the study under review) whose
    file names must equal the bank's, to verify a copied bank.
    """
    expect = expect or {}
    rep = {"rows": int(len(man)), "conditions": int(man["condition"].nunique()),
           "base_ids": int(man["base_id"].nunique()), "rec_ids": int(man["rec_id"].nunique()),
           "speech_share": float(man["label"].mean()), "axes": man["axis"].value_counts().to_dict(),
           "unknown_axis": int((man["axis"] == "unknown").sum()), "ids_hash": ids_hash(man["variant_id"])}
    failures = []
    for k, v in expect.items():
        got = rep.get(k)
        if got is None or (abs(got - v) > 1e-9 if isinstance(v, float) else got != v):
            failures.append(f"{k}: expected {v}, found {got}")
    if rep["unknown_axis"]:
        failures.append(f"{rep['unknown_axis']} file names without a recognisable condition")
    if names_csv is not None:
        ref = {Path(str(p)).stem for p in pd.read_csv(names_csv)["audio_path"]}
        have = set(man["variant_id"].astype(str))
        rep["names_missing"], rep["names_extra"] = len(ref - have), len(have - ref)
        if ref != have:
            failures.append(f"names differ from {Path(names_csv).name}: {len(ref - have)} missing, "
                            f"{len(have - ref)} extra")
    if check_audio:
        import soundfile as sf
        bad = []
        for p in man["path"]:
            try:
                info = sf.info(str(p))
                if info.samplerate != sr or abs(info.frames / info.samplerate - duration_s) > 0.5 / sr:
                    bad.append(f"{Path(p).name}: {info.samplerate} Hz, {info.frames} frames")
            except Exception as exc:
                bad.append(f"{Path(p).name}: {exc}")
        rep["audio_checked"], rep["audio_bad"] = int(len(man)), bad[:20]
        if bad:
            failures.append(f"{len(bad)} WAV files are not {duration_s:.3f} s at {sr} Hz")
    rep["failures"] = failures
    rep["ok"] = not failures
    return rep
