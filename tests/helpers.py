"""Shared fixtures for the infrastructure tests: tiny banks, a tiny adapter, a
fake Drive (a local folder) and a bare git repo that plays GitHub."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from erclm import banks  # noqa: E402
from erclm.storage import LocalRemote  # noqa: E402
from test_model_v2 import PROC, _randomize_lora, make_model  # noqa: E402

N = 32000


class SimulatedCrash(BaseException):
    """Stands for the VM dying. BaseException, so the runner cannot catch it."""


def _segment(rng, lab, k):
    t = np.arange(k) / 16000
    if lab:
        s = (np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)) * (1 + np.sin(2 * np.pi * 4 * t))
    else:
        s = rng.standard_normal(k)
    return 0.3 * s / np.abs(s).max()


def _variant(rng, lab, dur, snr=None):
    x = rng.standard_normal(N) * 1e-4
    k = int(dur * 16)
    seg = _segment(rng, lab, k)
    x[(N - k) // 2:(N - k) // 2 + k] += seg
    if snr is not None:
        x = x + rng.standard_normal(N) * np.sqrt(np.mean(seg ** 2)) / 10 ** (snr / 20)
    return x.astype(np.float32)


def write_test_bank(d: Path, n_rec_per_class=10) -> Path:
    """One-factor variants named as in the study under review: 3 durations, 2 SNRs and the neutral reverb and filter."""
    audio = d / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for lab in (1, 0):
        for r in range(n_rec_per_class):
            stem = f"voxconverse_rec{r:02d}b_0001" if lab else f"esc50_2-{2000 + r}-A-{r}_0001"
            variants = {f"dur{x}ms": _variant(rng, lab, x) for x in (20, 200, 1000)}
            for snr in (-10, 20):
                variants[f"snr{snr:+d}dB"] = _variant(rng, lab, 1000, snr)
            variants["reverbnone"] = variants["dur1000ms"]
            variants["filternone"] = variants["dur1000ms"]
            for tag, x in variants.items():
                name = f"{stem}_1000ms_{tag}.wav"
                sf.write(str(audio / name), x, 16000)
                rows.append({"audio_path": f"/mnt/fast/elsewhere/{name}",
                             "ground_truth": "SPEECH" if lab else "NONSPEECH"})
    pd.DataFrame(rows).to_csv(d / "test_metadata.csv", index=False)
    return d / "test_metadata.csv"


def write_train_bank(d: Path, n_rec_per_class=6) -> Path:
    """Factorial duration x SNR variants of separate recordings, like the LoRA manifest."""
    audio = d / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    rows = []
    for lab in (1, 0):
        for r in range(n_rec_per_class):
            stem = f"voxconverse_trn{r:02d}x_0001" if lab else f"esc50_3-{3000 + r}-A-{r}_0001"
            for dur in (200, 1000):
                for snr in (-10, 20):
                    name = f"{stem}_1000ms_dur{dur}ms_snr{snr:+d}dB.wav"
                    sf.write(str(audio / name), _variant(rng, lab, dur, snr), 16000)
                    rows.append({"audio_path": name, "ground_truth": "SPEECH" if lab else "NONSPEECH"})
    pd.DataFrame(rows).to_csv(d / "train_metadata.csv", index=False)
    return d / "train_metadata.csv"


def save_adapter(d: Path) -> Path:
    from peft import LoraConfig, get_peft_model
    pm = get_peft_model(make_model(PROC, seed=0),
                        LoraConfig(r=4, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    _randomize_lora(pm)
    pm.save_pretrained(str(d))
    return d


def git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def make_github(root: Path) -> Path:
    """Bare repo with a main branch and an orphan runs branch, the layout the runner publishes to."""
    bare = root / "github.git"
    git("init", "-q", "--bare", str(bare))
    seed = root / "seed"
    git("init", "-q", str(seed))
    ident = ["-c", "user.name=t", "-c", "user.email=t@t"]
    (seed / "README.md").write_text("code\n")
    git("add", "README.md", cwd=seed)
    git(*ident, "commit", "-q", "-m", "main", cwd=seed)
    git("push", "-q", str(bare), "HEAD:refs/heads/main", cwd=seed)
    git("checkout", "-q", "--orphan", "runs", cwd=seed)
    git("rm", "-q", "-rf", ".", cwd=seed)
    (seed / "README.md").write_text("runs\n")
    git("add", "README.md", cwd=seed)
    git(*ident, "commit", "-q", "-m", "runs", cwd=seed)
    git("push", "-q", str(bare), "HEAD:refs/heads/runs", cwd=seed)
    return bare


def show(bare: Path, path: str):
    """Content of a file on the runs branch, or None."""
    res = subprocess.run(["git", "--git-dir", str(bare), "show", f"runs:{path}"], capture_output=True)
    return res.stdout if res.returncode == 0 else None


def ls_runs(bare: Path, prefix: str = "") -> list:
    out = git("--git-dir", str(bare), "ls-tree", "-r", "--name-only", "runs")
    return [p for p in out.splitlines() if p.startswith(prefix)]


def build_world(root: Path, train: bool = True, n_rec_per_class: int = 10) -> dict:
    """Fake Drive with packed banks and the adapter, plus the fake GitHub."""
    remote = LocalRemote(root / "drive")
    src = root / "src_banks"
    test_csv = write_test_bank(src / "test", n_rec_per_class)
    tar, _ = banks.make_tar("test_bank_v1", test_csv, src / "test" / "audio", root / "packed")
    remote.put(tar, "data/test_bank_v1.tar")
    remote.put(tar.with_name(tar.name + ".sha256"), "data/test_bank_v1.tar.sha256")
    if train:
        tr_csv = write_train_bank(src / "train")
        tar, _ = banks.make_tar("train_bank_v1", tr_csv, src / "train" / "audio", root / "packed")
        remote.put(tar, "data/train_bank_v1.tar")
        remote.put(tar.with_name(tar.name + ".sha256"), "data/train_bank_v1.tar.sha256")
    ad = save_adapter(root / "adapter_src")
    for p in ad.iterdir():
        if p.is_file():
            remote.put(p, f"models/lora_csl_final/{p.name}")
    return {"remote": remote, "github": make_github(root), "root": root}


def tiny_config(train: bool = True, **over) -> dict:
    from erclm.runner import DEFAULTS, _merge
    cfg = {
        "run_name": "pilot-test",
        "model": {"revision": "test"},
        "llm_layers": [0, 2],
        "data": {"test": {"tar": "data/test_bank_v1.tar", "manifest": "test_bank_v1/manifest.csv",
                          "audio": "test_bank_v1/audio",
                          "expect": {"rows": 140, "conditions": 7, "base_ids": 20, "rec_ids": 20,
                                     "speech_share": 0.5}},
                 "train": ({"tar": "data/train_bank_v1.tar", "manifest": "train_bank_v1/manifest.csv",
                            "audio": "train_bank_v1/audio", "expect": {"rows": 48}} if train else None)},
        "selection": {"pilot": True, "n_per_class": 6, "seed": 42},
        "extraction": {"shard_size": 7, "n_content_free": 2, "log_every": 0},
        "analysis": {"bootstrap_B": 30, "prevalence_reps": 3},
        "heartbeat": {"log_every_s": 3600, "progress_every_s": 3600},
    }
    return _merge(_merge(DEFAULTS, cfg), over)


def tiny_loader():
    return make_model(PROC, seed=0), PROC
