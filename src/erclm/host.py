"""Host detection (Colab, Kaggle or local), working paths and secrets.

Secrets are read from environment variables first. On Colab the launcher copies
them from the notebook's secret store into the environment, because
google.colab.userdata only answers inside the notebook kernel and the runner
runs as a subprocess. The platform stores are tried as a fallback.

This module is called host and not platform so that it never shadows the
standard library module of that name.
"""
from __future__ import annotations

import base64
import binascii
import os
import platform as _platform
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SECRET_NAMES = ("GH_TOKEN", "RCLONE_CONF", "HF_TOKEN")

_WORK = {"colab": "/content/erclm_work", "kaggle": "/tmp/erclm_work"}


def detect() -> str:
    """'colab', 'kaggle' or 'local'. ERCLM_HOST overrides the guess."""
    forced = os.environ.get("ERCLM_HOST")
    if forced:
        return forced
    # Kaggle first: its image derives from Colab's and also sets COLAB_RELEASE_TAG
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"):
        return "colab"
    return "local"


def decode_rclone_conf(text: str) -> str:
    """RCLONE_CONF as pasted, or its base64 encoding.

    Kaggle's secret field may not keep line breaks, so the conf can be stored
    there as one base64 line. Plain text passes through unchanged.
    """
    t = (text or "").strip()
    if t and "\n" not in t and not t.startswith("["):
        try:
            dec = base64.b64decode(t, validate=True).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            return text
        dec = dec.lstrip("﻿")  # a byte order mark from Windows
        if dec.lstrip().startswith("["):
            return dec
    return text


def _platform_secret(host: str, name: str):
    try:
        if host == "colab":
            from google.colab import userdata  # type: ignore
            return userdata.get(name)
        if host == "kaggle":
            from kaggle_secrets import UserSecretsClient  # type: ignore
            return UserSecretsClient().get_secret(name)
    except Exception:
        return None
    return None


@dataclass
class Host:
    name: str
    work: Path

    @classmethod
    def current(cls, work=None) -> "Host":
        name = detect()
        root = Path(work or os.environ.get("ERCLM_WORK") or _WORK.get(name)
                    or Path.cwd() / "_work")
        root.mkdir(parents=True, exist_ok=True)
        return cls(name=name, work=root)

    def secret(self, name: str, required: bool = True):
        value = os.environ.get(name) or _platform_secret(self.name, name)
        if value:
            return value
        if required:
            raise RuntimeError(f"secret {name} is missing. Add it to the {self.name} secrets "
                               f"(see docs/running.md)")
        return None

    def secret_values(self) -> list:
        """Values of the known secrets present in the environment, for log redaction."""
        out = [v for v in (os.environ.get(n) for n in SECRET_NAMES) if v]
        conf = os.environ.get("RCLONE_CONF")
        if conf and decode_rclone_conf(conf) != conf:
            out.append(decode_rclone_conf(conf))
        return out

    def write_rclone_conf(self, text: str) -> Path:
        path = self.work / "rclone.conf"
        text = decode_rclone_conf(text)
        path.write_text(text if text.endswith("\n") else text + "\n")
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        return path


def gpu_info() -> dict:
    info = {"cuda": False}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = True
            info["cuda_version"] = torch.version.cuda
            info["devices"] = [{"name": torch.cuda.get_device_name(i),
                                "total_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)}
                               for i in range(torch.cuda.device_count())]
    except Exception as exc:  # torch missing on a CPU-only machine
        info["error"] = repr(exc)
    return info


def package_versions(names=("torch", "numpy", "pandas", "scipy", "sklearn", "transformers", "peft", "accelerate",
                            "bitsandbytes", "soundfile", "librosa", "yaml")) -> dict:
    out = {"python": sys.version.split()[0], "platform": _platform.platform()}
    for n in names:
        try:
            mod = __import__(n)
            out[n] = getattr(mod, "__version__", "?")
        except Exception:
            out[n] = None
    try:
        out["rclone"] = subprocess.run(["rclone", "version"], capture_output=True, text=True,
                                       timeout=30).stdout.splitlines()[0]
    except Exception:
        out["rclone"] = None
    return out


def code_sha(repo_dir) -> str:
    sha = os.environ.get("ERCLM_CODE_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30, check=True).stdout.strip()
    except Exception:
        return "unknown"
