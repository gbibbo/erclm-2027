"""ERC-LM experiment runner: resumable stages driven by a YAML config.

    python -m erclm.runner --config configs/pilot.yaml [--backbones base] [--force STAGE]

Stages, in order. GPU stages come first, so a cut during the CPU analysis never
costs GPU work.

    smoke              rclone puts, gets and verifies a test file (H2). Only when asked for
    data_test          bank copied, sha256 checked, counts and WAVs validated (H5)
    data_train         the same for the LoRA training bank, when configured
    check_<b>          model loads, the hooked state reproduces the logits, 2 s give
                       50 audio tokens, LoRA report (H4, H6). Needs no audio bank
    extract_<b>_eval   evaluation items plus content-free containers
    extract_<b>_train  LoRA training manifest, for the confirmatory heads
    analyze_<b>        readouts, tables, bootstrap and the score table (CPU)
    joint              Base + LoRA paired bootstrap from the score tables, H3 and H4 (CPU)

Where each artefact is stored (see docs/running.md).
    Drive  work/<run>/<b>/<eval|train>/shard_NNNNN.{npz,json,csv}   json is the receipt
           work/<run>/<b>/results/, work/<run>/joint/
           state/<run>/<stage>.started.json and .done.json
           logs/<run_id>/<session>.log
    runs   <run_id>/env.json, progress.json, results/, error.md

A stage that is marked done with the same configuration hash is skipped. An
extraction marked done or started with another configuration stops the run
with an explanation (use --force STAGE or a new run_name). Cheaper stages are
redone with a warning.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import io
import json
import os
import re
import secrets as _secrets
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import analysis, banks, export
from . import host as hostmod
from . import lib_v2 as L
from . import lib_v3
from .gitlog import NullRunsLog, RunsLog
from .storage import PART, LocalRemote, RcloneRemote, Remote, md5_file

REPO_ROOT = Path(__file__).resolve().parents[2]
lib_v3.register_prompts(L.PROMPTS)  # exploratory prompt variants; the confirmatory "ab" is untouched
BACKBONES = ("base", "lora")
NEUTRAL = ("dur_1000ms", "reverb_none", "filter_none")
PROBE_SEGMENT_MS = 200.0

DEFAULTS = {
    "run_name": None,
    "repo": "gbibbo/erclm-2027",
    "remote": "erclm:ERC-LM-2027",
    "model": {"id": "Qwen/Qwen2-Audio-7B-Instruct", "revision": None, "quant": "nf4"},
    "prompt_key": "ab",
    "llm_layers": [8, 16, 24],
    "backbones": ["base", "lora"],
    "adapter": "models/lora_csl_final",
    "data": {"test": None, "train": None},
    "selection": {"pilot": False, "n_per_class": 50, "seed": 42},
    "extraction": {"shard_size": 250, "n_content_free": 8, "n_head_checks": 3, "log_every": 100},
    "check": {"head_tol": 0.01},
    "analysis": {"bootstrap_B": 2000, "seed": 42, "h3_margin": 0.02, "prevalence_reps": 20},
    "heartbeat": {"log_every_s": 300, "progress_every_s": 1200},
    "stages": None,
    # exploratory interventions of 22 Oct (lib_v3): a transform of every container the model hears,
    # e.g. {kind: bandpass, lo: 60, hi: 6000, order: 8}, and the part of the model that keeps the
    # LoRA, e.g. audio_tower, language_model or language_model:16-31. Off by default
    "audio_transform": None,
    "lora_keep": None,
}


class StageConfigMismatch(RuntimeError):
    pass


class CheckFailed(RuntimeError):
    pass


# ============================================================================
# Configuration
# ============================================================================

def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as fh:
        cfg = _merge(DEFAULTS, yaml.safe_load(fh) or {})
    if not cfg.get("run_name"):
        raise ValueError(f"{path}: run_name is required")
    if not cfg["model"].get("revision"):
        raise ValueError(f"{path}: model.revision must pin a Hugging Face commit")
    return cfg


def config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _flatten(d, prefix=""):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: d}


def diff_specs(old: dict, new: dict) -> list:
    a, b = _flatten(old or {}), _flatten(new or {})
    return [f"{k}: {a.get(k)!r} -> {b.get(k)!r}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def run_id_for(cfg: dict, backbones=None) -> str:
    if backbones and list(backbones) != list(cfg["backbones"]):
        return f"{cfg['run_name']}-{'-'.join(backbones)}"
    return cfg["run_name"]


def new_session_id(host_name: str) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{host_name}-{_secrets.token_hex(2)}"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ============================================================================
# Logging with redaction
# ============================================================================

def secret_fragments(values) -> list:
    """Secrets to redact, including the tokens inside an rclone.conf."""
    out = []
    for v in values:
        if not v:
            continue
        out.append(v)
        out += re.findall(r'"(?:access_token|refresh_token)"\s*:\s*"([^"]+)"', v)
    return sorted({s for s in out if len(s) >= 8}, key=len, reverse=True)


class LogSink:
    def __init__(self, path: Path, secrets=()):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")
        self.lock = threading.Lock()
        self.secrets = secret_fragments(secrets)
        self.tail = collections.deque(maxlen=200)
        self._partial = ""

    def redact(self, s: str) -> str:
        for sec in self.secrets:
            s = s.replace(sec, "***")
        return s

    def write(self, s: str):
        with self.lock:
            self.fh.write(s)
            self.fh.flush()
            lines = (self._partial + s).split("\n")
            self._partial = lines.pop()
            self.tail.extend(lines)

    def tail_text(self) -> str:
        with self.lock:
            return "\n".join(list(self.tail) + ([self._partial] if self._partial else []))

    def close(self):
        with self.lock:
            self.fh.close()


class _Tee(io.TextIOBase):
    def __init__(self, stream, sink: LogSink):
        self.stream, self.sink = stream, sink

    def write(self, s):
        s = self.sink.redact(str(s))
        self.sink.write(s)
        try:
            self.stream.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass

    def isatty(self):
        return False


# ============================================================================
# Shards on the remote
# ============================================================================

SHARD_RE = re.compile(r"^shard_(\d{5})\.(npz|json|csv)$")


def _matches(listed: dict, receipt: Optional[dict]) -> bool:
    if not receipt or listed["size"] != receipt.get("size"):
        return False
    return not (listed.get("md5") and receipt.get("md5")) or listed["md5"] == receipt["md5"]


def sync_store(remote: Remote, rdir: str, ldir, fetch_npz: bool = False, log=print, warn=print) -> dict:
    """Mirror the committed shards of a remote store into ldir.

    A shard is committed when its CSV is on the remote. Its npz and CSV must
    match the size and MD5 of its receipt. A committed shard that fails this
    check is uncommitted (CSV deleted first) so its items are extracted again.
    Leftover .part files and npz or receipts without a CSV are removed, and so
    are local shard files that the remote does not hold.
    """
    ldir = Path(ldir)
    ldir.mkdir(parents=True, exist_ok=True)
    remote.dedupe(rdir)
    listing = remote.ls(rdir)
    for name in [n for n in listing if n.endswith(PART)]:
        remote.delete(f"{rdir}/{name}")
    shards = {}
    for name, info in listing.items():
        m = SHARD_RE.match(name)
        if m:
            shards.setdefault(m.group(1), {})[m.group(2)] = info
    tmp = ldir / ".receipts"
    receipts = [f"shard_{k}.json" for k, f in shards.items() if "csv" in f and "json" in f]
    remote.get_many(rdir, receipts, tmp, listing)
    good, dropped = [], []
    for k, f in sorted(shards.items()):
        if "csv" not in f:
            continue
        reason = None
        if "npz" not in f or "json" not in f:
            reason = "npz or receipt missing"
        else:
            try:
                rec = json.loads((tmp / f"shard_{k}.json").read_text())
            except Exception:
                reason = "unreadable receipt"
            else:
                if not _matches(f["npz"], rec.get("npz")):
                    reason = "npz does not match its receipt (truncated or corrupt)"
                elif not _matches(f["csv"], rec.get("csv")):
                    reason = "csv does not match its receipt"
        (dropped if reason else good).append((k, reason) if reason else k)
    for k, reason in dropped:
        warn(f"{rdir}/shard_{k}: {reason}. Uncommitting it, its items will be extracted again")
        for ext in ("csv", "npz", "json"):
            remote.delete(f"{rdir}/shard_{k}.{ext}")
    orphans = [k for k, f in shards.items() if "csv" not in f]
    for k in orphans:
        for ext in ("npz", "json"):
            remote.delete(f"{rdir}/shard_{k}.{ext}")
    keep = set(good)
    for p in ldir.glob("shard_*"):
        m = SHARD_RE.match(p.name)
        if not m or m.group(1) not in keep:
            p.unlink()
    names = [f"shard_{k}.csv" for k in good] + ([f"shard_{k}.npz" for k in good] if fetch_npz else [])
    remote.get_many(rdir, names, ldir, listing)
    shutil.rmtree(tmp, ignore_errors=True)
    rep = {"committed": len(good), "dropped": [f"shard_{k}: {r}" for k, r in dropped],
           "orphans_removed": len(orphans)}
    log(f"{rdir}: {len(good)} committed shards, {len(dropped)} dropped, {len(orphans)} orphans removed")
    return rep


class SyncedFeatureStore(L.FeatureStore):
    """FeatureStore that uploads every shard right after it is written.

    Upload order is npz, receipt, CSV. The CSV on the remote commits the shard.
    Every row also records the session and host that extracted it, because FP16
    results differ slightly between GPUs and CUDA versions.
    """

    def __init__(self, out_dir, remote: Remote, remote_dir: str, shard_size: int = 250,
                 on_add: Optional[Callable] = None, tags: Optional[dict] = None):
        super().__init__(out_dir, shard_size=shard_size)
        self.remote, self.rdir, self.on_add, self.tags = remote, remote_dir, on_add, dict(tags or {})

    def add(self, meta: dict, arrays: dict):
        super().add({**meta, **self.tags}, arrays)
        if self.on_add:
            self.on_add()

    def flush(self):
        if not self._meta:
            return
        before = set(self.dir.glob("shard_*.csv"))
        super().flush()
        for csv in sorted(set(self.dir.glob("shard_*.csv")) - before):
            self.upload(csv)

    def upload(self, csv: Path):
        npz, receipt = csv.with_suffix(".npz"), csv.with_suffix(".json")
        rec = {"npz": self.remote.put(npz, f"{self.rdir}/{npz.name}"),
               "csv": {"size": csv.stat().st_size, "md5": md5_file(csv)},
               "n_items": int(len(pd.read_csv(csv, usecols=["variant_id"]))), "written_at": _now()}
        receipt.write_text(json.dumps(rec, indent=2))
        self.remote.put(receipt, f"{self.rdir}/{receipt.name}")
        self.remote.put(csv, f"{self.rdir}/{csv.name}")


# ============================================================================
# Model
# ============================================================================

def load_hf(cfg: dict, hf_token: Optional[str] = None):
    """Qwen2-Audio at the pinned revision, NF4 with double quantisation and FP16 compute."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration
    m = cfg["model"]
    kw_hub = {"revision": m["revision"], "token": hf_token}
    processor = AutoProcessor.from_pretrained(m["id"], **kw_hub)
    if m["quant"] in ("nf4", "fp4"):
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=m["quant"],
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=m["quant"] == "nf4")
        # fp16 for the modules bitsandbytes leaves unquantised too: the checkpoint is bf16,
        # transformers 5 keeps it by default, and the T4 has no native bf16
        kw = dict(quantization_config=qc, dtype=torch.float16, device_map="auto")
    else:
        kw = dict(dtype=torch.float16, device_map="auto")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(m["id"], **kw_hub, **kw).eval()
    return model, processor


def vram() -> dict:
    try:
        import torch
        if torch.cuda.is_available():  # summed over devices; device_map="auto" splits the model on 2 x T4
            n = range(torch.cuda.device_count())
            out = {"allocated_gb": round(sum(torch.cuda.memory_allocated(i) for i in n) / 2**30, 3),
                   "max_allocated_gb": round(sum(torch.cuda.max_memory_allocated(i) for i in n) / 2**30, 3),
                   "reserved_gb": round(sum(torch.cuda.memory_reserved(i) for i in n) / 2**30, 3)}
            if len(n) > 1:
                out["per_device_allocated_gb"] = [round(torch.cuda.memory_allocated(i) / 2**30, 3) for i in n]
            return out
    except Exception:
        pass
    return {}


def probe_wav() -> np.ndarray:
    """Fixed 2 s container with a centred 200 ms harmonic burst, for the model checks."""
    rng = np.random.default_rng(20261015)
    n = int(L.SR * L.CONTAINER_MS / 1000)
    x = rng.standard_normal(n) * L.PAD_SIGMA
    k = int(L.SR * PROBE_SEGMENT_MS / 1000)
    t = np.arange(k) / L.SR
    seg = np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)
    x[(n - k) // 2:(n - k) // 2 + k] += 0.3 * seg / np.abs(seg).max()
    return x.astype(np.float32)


@dataclass
class Bundle:
    backbone: str
    model: object
    processor: object
    recorder: object
    labels: dict
    audio_id: int
    lora_rep: Optional[dict]


class Models:
    """Loads the base model once and attaches the adapter for the LoRA backbone."""

    def __init__(self, runner: "Runner"):
        self.r = runner
        self.model = self.processor = self.bundle = None
        self.wrapped = False

    def _load(self):
        self.release()
        t0 = time.time()
        self.model, self.processor = self.r.loader()
        self.wrapped = False
        self.r.model_info = {"load_s": round(time.time() - t0, 1), "vram_after_load": vram()}
        self.r.log(f"model loaded in {time.time() - t0:.0f} s, VRAM {self.r.model_info['vram_after_load']}")

    def release(self):
        if self.bundle is not None:
            self.bundle.recorder.remove()
        self.model = self.processor = self.bundle = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def get(self, backbone: str) -> Bundle:
        if self.bundle is not None and self.bundle.backbone == backbone:
            return self.bundle
        if self.bundle is not None:
            self.bundle.recorder.remove()
            self.bundle = None
        rep = None
        if backbone == "base":
            if self.model is None or self.wrapped:
                self._load()
            model = self.model
        elif backbone == "lora":
            if self.model is None or self.wrapped:
                self._load()
            model, rep = L.load_lora(self.model, self.r.fetch_adapter())
            if self.r.cfg.get("lora_keep"):
                rep = {**rep, "restricted": lib_v3.restrict_lora(model, self.r.cfg["lora_keep"])}
            self.model, self.wrapped = model, True
            self.r.log(f"LoRA attached: {rep}")
        else:
            raise ValueError(f"unknown backbone {backbone}")
        proc = self.processor
        rec = L.Recorder(model, llm_layers=self.r.cfg["llm_layers"])
        labels = L.label_token_info(proc.tokenizer, L.PROMPTS[self.r.cfg["prompt_key"]]["labels"])
        aid = proc.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.bundle = Bundle(backbone, model, proc, rec, labels, aid, rep)
        return self.bundle


# ============================================================================
# Progress and heartbeat
# ============================================================================

class Progress:
    def __init__(self):
        self.lock = threading.Lock()
        self.stages = {}
        self.current = None
        self.warnings = []
        self._t = {}

    def set(self, name: str, **kw):
        with self.lock:
            self.stages.setdefault(name, {"status": "pending"}).update(kw)

    def start_items(self, name: str, total: int, done: int):
        with self.lock:
            self.stages.setdefault(name, {}).update(total=total, done=done, done_at_start=done)
            self._t[name] = time.time()

    def item(self, name: str):
        with self.lock:
            st = self.stages[name]
            st["done"] = st.get("done", 0) + 1
            new = st["done"] - st.get("done_at_start", 0)
            el = time.time() - self._t.get(name, time.time())
            if new > 0:
                st["s_per_item"] = round(el / new, 4)
                st["eta_min"] = round(max(st.get("total", 0) - st["done"], 0) * el / new / 60, 1)

    def warn(self, msg: str):
        with self.lock:
            if msg not in self.warnings:
                self.warnings = (self.warnings + [msg])[-50:]

    def snapshot(self) -> dict:
        with self.lock:
            return {"current_stage": self.current, "stages": copy.deepcopy(self.stages),
                    "warnings": list(self.warnings)}


class Heartbeat(threading.Thread):
    def __init__(self, runner: "Runner", log_every_s: float, progress_every_s: float):
        super().__init__(daemon=True, name="erclm-heartbeat")
        self.r, self.log_every, self.progress_every = runner, log_every_s, progress_every_s
        self._halt = threading.Event()

    def run(self):
        last_log = last_prog = time.time()
        tick = max(0.05, min(5.0, self.log_every, self.progress_every))
        while not self._halt.wait(tick):
            now = time.time()
            try:
                if now - last_log >= self.log_every:
                    self.r.upload_log()
                    last_log = now
                if now - last_prog >= self.progress_every:
                    self.r.publish_progress()
                    last_prog = now
            except Exception as exc:
                self.r.warn(f"heartbeat: {exc}")

    def stop(self):
        self._halt.set()
        self.join(timeout=120)


# ============================================================================
# Runner
# ============================================================================

@dataclass(frozen=True)
class Stage:
    name: str
    kind: str
    backbone: Optional[str] = None
    which: Optional[str] = None

    @property
    def on_mismatch(self) -> str:
        return "stop" if self.kind in ("extract", "omni") else "redo"


class Runner:
    def __init__(self, cfg: dict, host: hostmod.Host, remote: Remote, runs, model_loader=None,
                 backbones=None, only=None, force=(), session_id=None, run_id=None,
                 config_file: str = "", repo_dir=REPO_ROOT, omni_loader=None):
        self.cfg, self.host, self.remote, self.runs = cfg, host, remote, runs
        self.loader = model_loader or (lambda: load_hf(cfg, host.secret("HF_TOKEN", required=False)))
        self.omni_loader = omni_loader or self._load_omni
        self.backbones = list(backbones or cfg["backbones"])
        only = only or cfg.get("stages")
        self.only = set(only) if only else None
        self.force = set(force or ())
        self.session_id = session_id or new_session_id(host.name)
        self.run_name = cfg["run_name"]
        self.run_id = run_id or run_id_for(cfg, backbones)
        self.config_file = config_file
        self.code_sha = hostmod.code_sha(repo_dir)
        self.work = Path(host.work)
        self.progress = Progress()
        self.models = Models(self)
        self.model_info = {}
        self.env = {}
        self.sink = None
        self._memo = {}
        self._log_lock = threading.Lock()

    # ------------------------------------------------------------ output
    def log(self, msg: str):
        print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)

    def warn(self, msg: str):
        self.progress.warn(msg)
        self.log(f"WARNING {msg}")

    def publish(self, files: dict, message: str, delete=()) -> bool:
        try:
            self.runs.publish(files, delete=delete, message=message)
            return True
        except Exception as exc:
            self.warn(f"could not publish to runs: {exc}")
            return False

    def publish_progress(self):
        snap = {"run_id": self.run_id, "session_id": self.session_id, "host": self.host.name,
                "code_sha": self.code_sha, "updated_at": _now(), **self.progress.snapshot()}
        self.publish({"progress.json": json.dumps(snap, indent=2, default=str)}, f"{self.run_id}: progress")

    def upload_log(self):
        if self.sink is None:
            return
        with self._log_lock:
            snap = self.sink.path.with_name(self.sink.path.stem + ".snapshot.log")
            with self.sink.lock:
                shutil.copyfile(self.sink.path, snap)
            self.remote.put(snap, f"logs/{self.run_id}/{self.session_id}.log")

    # ------------------------------------------------------------ paths
    def r_state(self, stage: str, kind: str) -> str:
        return f"state/{self.run_name}/{stage}.{kind}.json"

    def r_store(self, b: str, which: str) -> str:
        return f"work/{self.run_name}/{b}/{which}"

    def l_store(self, b: str, which: str) -> Path:
        return self.work / "work" / self.run_name / b / which

    def r_results(self, b: str) -> str:
        return f"work/{self.run_name}/{b}/results"

    # ------------------------------------------------------------ plan
    def universe(self) -> dict:
        has_train = bool(self.cfg["data"].get("train"))
        st = [Stage("smoke", "smoke")]
        if self.cfg.get("banks"):
            st.append(Stage("build_banks", "banks"))
        if self.cfg.get("provenance"):
            st.append(Stage("build_prov", "prov"))
        if self.cfg.get("retrain"):
            st.append(Stage("train_lora", "train"))
        st.append(Stage("data_test", "data", which="test"))
        if has_train:
            st.append(Stage("data_train", "data", which="train"))
        if self.cfg.get("omni"):
            st.append(Stage("extract_omni", "omni"))
        for b in BACKBONES:
            st += [Stage(f"check_{b}", "check", b), Stage(f"extract_{b}_eval", "extract", b, "eval")]
            if has_train:
                st.append(Stage(f"extract_{b}_train", "extract", b, "train"))
        st += [Stage(f"analyze_{b}", "analyze", b) for b in BACKBONES]
        st.append(Stage("joint", "joint"))
        return {s.name: s for s in st}

    def plan(self) -> list:
        U = self.universe()
        if self.only:
            unknown = self.only - set(U)
            if unknown:
                raise ValueError(f"unknown stages {sorted(unknown)}. Known: {list(U)}")
            return [s for s in U.values() if s.name in self.only]
        out = []
        for s in U.values():
            if s.kind in ("smoke", "banks", "prov", "train", "omni"):  # only when a config or --stages asks for them
                continue
            if s.kind == "data" and not self.cfg["data"].get(s.which):
                continue
            if s.backbone and s.backbone not in self.backbones:
                continue
            if s.kind == "joint":
                others = [b for b in BACKBONES if b not in self.backbones]
                if any(not self.remote.exists(self.r_state(f"analyze_{b}", "done")) for b in others):
                    continue
            out.append(s)
        return out

    # ------------------------------------------------------------ specs
    def adapter_fingerprint(self) -> dict:
        if "adapter" not in self._memo:
            listing = self.remote.ls(self.cfg["adapter"])
            if not listing:
                raise FileNotFoundError(f"adapter folder {self.cfg['adapter']} is missing or empty on the remote")
            self._memo["adapter"] = {"path": self.cfg["adapter"],
                                     "files": {n: i.get("md5") or i["size"] for n, i in listing.items()}}
        return self._memo["adapter"]

    def spec(self, s: Stage) -> dict:
        key = ("spec", s.name)
        if key in self._memo:
            return self._memo[key]
        c, U = self.cfg, self.universe()
        model = {"id": c["model"]["id"], "revision": c["model"]["revision"], "quant": c["model"]["quant"],
                 "prompt_key": c["prompt_key"]}
        if s.kind == "smoke":
            spec = {}
        elif s.kind == "banks":
            from . import bankgen
            b = c["banks"]
            spec = {"protocol": bankgen.PROTOCOL_VERSION, "seed": b.get("seed", 42), "names": b["names"],
                    "counts": b.get("counts") or bankgen.DEFAULT_COUNTS, "sources": b["sources"],
                    "expect": b.get("expect", {}),
                    "code": hashlib.sha256(Path(bankgen.__file__).read_bytes()).hexdigest()[:16]}
        elif s.kind == "prov":
            from . import bankgen, bankgen_v3
            p = c["provenance"]
            code = Path(bankgen_v3.__file__).read_bytes() + Path(bankgen.__file__).read_bytes()
            spec = {"protocol": bankgen_v3.PROTOCOL, "bankgen": bankgen.PROTOCOL_VERSION, "seed": p.get("seed", 42),
                    "name": p["name"], "per_recording": p.get("per_recording", 1),
                    "min_recordings": p.get("min_recordings", 1), "exclude_clips": p["exclude_clips"],
                    "sources": p["sources"], "code": hashlib.sha256(code).hexdigest()[:16]}
        elif s.kind == "train":
            from . import lora_train
            r = c["retrain"]
            spec = {"model": model, "recipe": {**lora_train.RECIPE, **(r.get("recipe") or {})},
                    "max_steps": r.get("max_steps", -1), "output": r["output"],
                    "train_sha256": banks.remote_sha(self.remote, c["data"]["train"]["tar"]),
                    "dev_sha256": banks.remote_sha(self.remote, c["data"]["dev"]["tar"]),
                    "code": hashlib.sha256(Path(lora_train.__file__).read_bytes()).hexdigest()[:16]}
        elif s.kind == "data":
            d = c["data"].get(s.which)
            if not d:
                raise ValueError(f"data.{s.which} is not configured")
            spec = {"tar": d["tar"], "sha256": banks.remote_sha(self.remote, d["tar"]), "manifest": d["manifest"],
                    "audio": d["audio"], "expect": d.get("expect") or {}, "names_csv": d.get("names_csv")}
        elif s.kind == "check":
            spec = {"model": model, "llm_layers": list(c["llm_layers"]), "head_tol": c["check"]["head_tol"],
                    "adapter": self.adapter_fingerprint() if s.backbone == "lora" else None}
            if s.backbone == "lora" and c.get("lora_keep"):
                spec["lora_keep"] = c["lora_keep"]
        elif s.kind == "extract":
            bank = "test" if s.which == "eval" else "train"
            spec = {"model": model, "backbone": s.backbone, "llm_layers": list(c["llm_layers"]),
                    "adapter": self.adapter_fingerprint() if s.backbone == "lora" else None,
                    "bank_sha256": self.spec(U[f"data_{bank}"])["sha256"]}
            if s.which == "eval":
                spec.update(selection=c["selection"], n_content_free=c["extraction"]["n_content_free"])
            # only when set, so the specs of earlier runs keep their hashes
            if s.backbone == "lora" and c.get("lora_keep"):
                spec["lora_keep"] = c["lora_keep"]
            if c.get("audio_transform"):
                spec["audio_transform"] = c["audio_transform"]
            if c.get("generate"):
                spec["generate"] = c["generate"]
        elif s.kind == "omni":
            from . import omni
            o = c["omni"]
            spec = {"model": {k: c["model"][k] for k in ("id", "revision", "quant")},
                    "compute_dtype": o.get("compute_dtype", "float16"), "tracks": omni.tracks(o),
                    "conditions": list(o["conditions"]), "selection": c["selection"], "pip": c.get("pip"),
                    "bank_sha256": self.spec(U["data_test"])["sha256"],
                    "code": hashlib.sha256(Path(omni.__file__).read_bytes()).hexdigest()[:16]}
        elif s.kind == "analyze":
            ups = [n for n in (f"extract_{s.backbone}_eval", f"extract_{s.backbone}_train") if n in U]
            spec = {"extract": {n: self.spec_hash(U[n]) for n in ups},
                    "analysis": {k: v for k, v in c["analysis"].items() if k != "h3_margin"}}
        elif s.kind == "joint":
            spec = {"analyze": {f"analyze_{b}": self.spec_hash(U[f"analyze_{b}"]) for b in BACKBONES},
                    "analysis": c["analysis"]}
        else:
            raise ValueError(f"unknown stage kind {s.kind}")
        self._memo[key] = spec
        return spec

    def spec_hash(self, s: Stage) -> str:
        return config_hash(self.spec(s))

    # ------------------------------------------------------------ resources
    def bank(self, which: str):
        key = ("bank", which)
        if key not in self._memo:
            spec = self.cfg["data"].get(which)
            if not spec:
                raise ValueError(f"data.{which} is not configured")
            self._memo[key] = banks.fetch(self.remote, spec, self.work / "data",
                                          role="test" if which == "test" else "train", log=self.log)
        return self._memo[key]

    def eval_manifest(self) -> pd.DataFrame:
        man = self.bank("test")[0]
        sel = self.cfg["selection"]
        if sel.get("pilot"):
            man = L.select_pilot(man, sel["n_per_class"], sel["seed"])
        if sel.get("axes"):
            man = man[man["axis"].isin(sel["axes"])].reset_index(drop=True)
        return man

    def content_free(self) -> pd.DataFrame:
        if "cf" not in self._memo:
            self._memo["cf"] = L.make_content_free(self.work / "content_free",
                                                   n_noise_seeds=self.cfg["extraction"]["n_content_free"])
        return self._memo["cf"]

    def fetch_adapter(self) -> Path:
        local = self.work / "adapter" / Path(self.cfg["adapter"]).name
        self.remote.get_dir(self.cfg["adapter"], local)
        return local

    # ------------------------------------------------------------ stages
    def _reset(self, s: Stage, why: str):
        self.log(f"[{s.name}] resetting: {why}")
        if s.kind == "extract":
            self.remote.purge(self.r_store(s.backbone, s.which))
            shutil.rmtree(self.l_store(s.backbone, s.which), ignore_errors=True)
        elif s.kind == "omni":
            from . import omni
            for t in omni.tracks(self.cfg["omni"]):
                self.remote.purge(self.r_store("omni", t["name"]))
                shutil.rmtree(self.l_store("omni", t["name"]), ignore_errors=True)
        elif s.kind == "analyze":
            self.remote.purge(self.r_results(s.backbone))
        elif s.kind == "joint":
            self.remote.purge(f"work/{self.run_name}/joint")
        self.remote.delete(self.r_state(s.name, "done"))
        self.remote.delete(self.r_state(s.name, "started"))

    def _mismatch(self, s: Stage, what: str, old_spec: dict, spec: dict):
        diff = diff_specs(old_spec, spec)
        msg = f"[{s.name}] {what} with another configuration. Differences: {diff}"
        if s.on_mismatch == "stop":
            raise StageConfigMismatch(msg + f". Nothing was skipped or overwritten. To redo it, relaunch with "
                                            f"--force {s.name}. For a new run, change run_name in the config")
        self.warn(msg + ". Redoing it")
        self._reset(s, "configuration changed")

    def execute(self, s: Stage):
        self.progress.current = s.name
        if s.kind == "smoke":  # a connectivity test, run every time and never marked
            result, files = self.stage_smoke(s)
            self.publish(files, f"{self.run_id}: smoke test")
            self.progress.set(s.name, status="done")
            return {"result": result}
        if s.name in self.force or "all" in self.force:
            self._reset(s, "--force")
        spec, h = self.spec(s), self.spec_hash(s)
        done = self.remote.get_json(self.r_state(s.name, "done"))
        if done is not None and done.get("config_hash") == h and s.kind == "extract" \
                and not self._verify_extract(s, done):
            self.warn(f"[{s.name}] is marked done but its committed shards no longer cover its items. "
                      f"Extracting the missing ones")
            self.remote.delete(self.r_state(s.name, "done"))
            done = None
        if done is not None:
            if done.get("config_hash") == h:
                self.log(f"[{s.name}] already done on {done.get('finished_at')} by {done.get('session_id')}, skipping")
                self.progress.set(s.name, status="done", skipped=True, finished_at=done.get("finished_at"))
                if not done.get("published", True):
                    self._republish(s, done)
                return done
            self._mismatch(s, "is marked done", done.get("config"), spec)
        started = self.remote.get_json(self.r_state(s.name, "started"))
        if started is not None and started.get("config_hash") != h:
            self._mismatch(s, "was started", started.get("config"), spec)
            started = None
        if started is None:
            self.remote.put_json({"stage": s.name, "config_hash": h, "config": spec, "session_id": self.session_id,
                                  "code_sha": self.code_sha, "started_at": _now()}, self.r_state(s.name, "started"))
        self.progress.set(s.name, status="running", started_at=_now())
        self.publish_progress()
        t0 = time.time()
        result, files = getattr(self, f"stage_{s.kind}")(s)
        published = self.publish(files, f"{self.run_id}: {s.name} done") if files else True
        gpu = hostmod.gpu_info().get("devices")
        marker = {"stage": s.name, "config_hash": h, "config": spec, "code_sha": self.code_sha,
                  "session_id": self.session_id, "host": self.host.name, "gpu": gpu, "finished_at": _now(),
                  "elapsed_s": round(time.time() - t0, 1), "published": published, "result": result}
        self.remote.put_json(marker, self.r_state(s.name, "done"))
        self.progress.set(s.name, status="done", elapsed_s=marker["elapsed_s"])
        self.publish_progress()
        return marker

    def _republish(self, s: Stage, marker: dict):
        files = {}
        if s.kind in ("analyze", "joint"):
            rdir = self.r_results(s.backbone) if s.kind == "analyze" else f"work/{self.run_name}/joint"
            local = self.work / "republish" / s.name
            self.remote.get_dir(rdir, local)
            sub = s.backbone if s.kind == "analyze" else "joint"
            files = {f"results/{sub}/{p.name}": p for p in sorted(local.iterdir()) if p.is_file()}
        else:
            files = {f"{s.name}.json": json.dumps(marker.get("result"), indent=2, default=str)}
        if self.publish(files, f"{self.run_id}: {s.name} republished"):
            marker["published"] = True
            self.remote.put_json(marker, self.r_state(s.name, "done"))

    def stage_smoke(self, s: Stage):
        payload = os.urandom(8 << 20)
        rel = f"state/smoke/{self.session_id}.bin"
        t0 = time.time()
        info = self.remote.put_bytes(payload, rel)
        t1 = time.time()
        back = self.remote.get_bytes(rel)
        t2 = time.time()
        self.remote.delete(rel)
        if back != payload:
            raise RuntimeError("smoke test: the downloaded file differs from the uploaded one")
        res = {"bytes": len(payload), "md5": info["md5"], "upload_s": round(t1 - t0, 2),
               "download_s": round(t2 - t1, 2), "remote": self.remote.describe()}
        self.log(f"smoke test passed: {res}")
        return res, {"smoke.json": json.dumps(res, indent=2)}

    def stage_banks(self, s: Stage):
        """Rebuild the test, train and dev banks (docs/banks.md) and store them on Drive.

        A bank whose tar, sha256 and protocol record are already on the remote with the same
        specification is not rebuilt, so a cut costs at most the bank in progress.
        """
        from . import bankgen
        b, spec = self.cfg["banks"], self.spec(s)
        tag = config_hash(spec)
        rooms_rel = f"data/rirs/rooms_{b['names']['test'].split('_')[-1]}.npz"
        out = self.work / "banks"
        out.mkdir(parents=True, exist_ok=True)
        local_rooms = out / Path(rooms_rel).name
        if self.remote.exists(rooms_rel) and self.remote.exists(rooms_rel[:-4] + ".json"):
            self.remote.get(rooms_rel, local_rooms)
            self.remote.get(rooms_rel[:-4] + ".json", local_rooms.with_suffix(".json"))
            rooms = bankgen.load_rooms(local_rooms)
            self.log(f"rooms loaded from {rooms_rel}")
        else:
            rooms = bankgen.make_rooms(seed=spec["seed"], log=self.log)
            bankgen.save_rooms(rooms, local_rooms)
            self.remote.put(local_rooms, rooms_rel)
            self.remote.put(local_rooms.with_suffix(".json"), rooms_rel[:-4] + ".json")
        vox, esc, esc_paths = bankgen.fetch_sources(self.work / "sources", b["sources"], log=self.log)
        clips = bankgen.select_clips(vox, esc, seed=spec["seed"], counts=spec["counts"], log=self.log)
        clips_csv = out / "clips.csv"
        clips.to_csv(clips_csv, index=False)
        self.remote.put(clips_csv, f"data/manifests/clips_{tag}.csv")
        files, result = {"banks/clips.csv": clips_csv,
                         "banks/rooms.json": local_rooms.with_suffix(".json")}, {"clips": int(len(clips))}
        for split, conds in (("test", bankgen.test_conditions()), ("train", bankgen.factorial_conditions()),
                             ("dev", bankgen.factorial_conditions())):
            name = b["names"][split]
            record = self.remote.get_json(f"data/manifests/{name}.protocol.json")
            if record and record.get("spec_hash") == tag and self.remote.exists(f"data/{name}.tar.sha256"):
                self.log(f"{name} already built with this specification, skipping")
                result[split] = record["report"]
                continue
            summary = bankgen.build_bank(name, clips[clips.split == split], conds, vox, esc_paths, rooms, out,
                                         log=self.log)
            self.log(f"{name}: {summary['variants']} variants rendered, validating")
            man = L.load_manifest(out / name / "manifest.csv", audio_root=out / name / "audio",
                                  role="test" if split == "test" else "train")
            report = {**summary, **banks.validate(man, spec["expect"].get(split), check_audio=True)}
            report.pop("audio_bad", None)
            if not report["ok"]:
                self.publish({f"banks/{name}.json": json.dumps(report, indent=2, default=str)},
                             f"{self.run_id}: {name} failed validation")
                raise RuntimeError(f"{name} failed validation: {report['failures']}")
            self.log(f"{name}: valid, packing")
            tar, sha = bankgen.pack(name, out)
            self.log(f"{name}: tar of {tar.stat().st_size / 1e9:.2f} GB, uploading")
            self.remote.put(tar, f"data/{name}.tar")
            self.remote.put(tar.with_name(tar.name + ".sha256"), f"data/{name}.tar.sha256")
            meta_gz = out / f"{name}.metadata.csv.gz"
            pd.read_csv(out / name / "metadata.csv").to_csv(meta_gz, index=False)
            self.remote.put(meta_gz, f"data/manifests/{name}.metadata.csv.gz")
            report["sha256"] = sha
            self.remote.put_json({"spec_hash": tag, "spec": spec, "report": report},
                                 f"data/manifests/{name}.protocol.json")
            files[f"banks/{name}.json"] = json.dumps(report, indent=2, default=str)
            files[f"banks/{name}.metadata.csv.gz"] = meta_gz.read_bytes()
            result[split] = report
            shutil.rmtree(out / name, ignore_errors=True)
            tar.unlink()
            self.log(f"{name}: {report['rows']} variants, sha256 {sha[:16]}")
        return result, files

    def stage_prov(self, s: Stage):
        """Build the provenance bank (erclm.bankgen_v3, decision D2 in docs/exploratory_log.md):
        speech and non-speech from the same VoxConverse recordings, none used by the v2 banks.
        A bank already on the remote with the same specification is not rebuilt."""
        from . import bankgen, bankgen_v3
        p, spec = self.cfg["provenance"], self.spec(s)
        name, tag = p["name"], config_hash(spec)
        record = self.remote.get_json(f"data/manifests/{name}.protocol.json")
        if record and record.get("spec_hash") == tag and self.remote.exists(f"data/{name}.tar.sha256"):
            self.log(f"{name} already built with this specification, skipping")
            return record["report"], {}
        out = self.work / "banks"
        out.mkdir(parents=True, exist_ok=True)
        excl_csv = out / Path(p["exclude_clips"]).name
        self.remote.get(p["exclude_clips"], excl_csv)
        excl = pd.read_csv(excl_csv)
        exclude = set(excl.loc[excl["dataset"] == "voxconverse", "recording"].astype(str))
        vox, _, _ = bankgen.fetch_sources(self.work / "sources", p["sources"], log=self.log)
        clips = bankgen_v3.select_provenance_clips(vox, exclude, seed=spec["seed"],
                                                   per_recording=spec["per_recording"], log=self.log)
        n_rec = int(clips["recording"].nunique())
        if n_rec < spec["min_recordings"]:
            raise RuntimeError(f"{name}: only {n_rec} recordings with both classes, "
                               f"fewer than the {spec['min_recordings']} required")
        clips_csv = out / f"{name}.clips.csv"
        clips.to_csv(clips_csv, index=False)
        self.remote.put(clips_csv, f"data/manifests/{name}.clips.csv")
        conds = bankgen_v3.duration_conditions()
        summary = bankgen.build_bank(name, clips, conds, vox, {}, None, out, log=self.log)
        man = L.load_manifest(out / name / "manifest.csv", audio_root=out / name / "audio", role="test")
        expect = {"rows": len(clips) * len(conds), "conditions": len(conds), "base_ids": len(clips),
                  "rec_ids": n_rec, "speech_share": 0.5}
        report = {**summary, **banks.validate(man, expect, check_audio=True)}
        report.pop("audio_bad", None)
        files = {f"banks/{name}.clips.csv": clips_csv}
        if not report["ok"]:
            self.publish({f"banks/{name}.json": json.dumps(report, indent=2, default=str)},
                         f"{self.run_id}: {name} failed validation")
            raise RuntimeError(f"{name} failed validation: {report['failures']}")
        tar, sha = bankgen.pack(name, out)
        self.log(f"{name}: tar of {tar.stat().st_size / 1e9:.2f} GB, uploading")
        self.remote.put(tar, f"data/{name}.tar")
        self.remote.put(tar.with_name(tar.name + ".sha256"), f"data/{name}.tar.sha256")
        meta_gz = out / f"{name}.metadata.csv.gz"
        pd.read_csv(out / name / "metadata.csv").to_csv(meta_gz, index=False)
        self.remote.put(meta_gz, f"data/manifests/{name}.metadata.csv.gz")
        report["sha256"] = sha
        self.remote.put_json({"spec_hash": tag, "spec": spec, "report": report},
                             f"data/manifests/{name}.protocol.json")
        files[f"banks/{name}.json"] = json.dumps(report, indent=2, default=str)
        files[f"banks/{name}.metadata.csv.gz"] = meta_gz.read_bytes()
        shutil.rmtree(out / name, ignore_errors=True)
        tar.unlink()
        self.log(f"{name}: {report['rows']} variants from {n_rec} recordings, sha256 {sha[:16]}")
        return report, files

    def stage_train(self, s: Stage):
        """Retrain the task LoRA with the recipe of the study under review (erclm.lora_train) on the
        train bank, selecting
        on the dev bank. Checkpoints go to checkpoints/<run_name>/ and the chosen adapter to the
        configured output folder."""
        from . import lora_train
        spec = self.spec(s)
        train_man = self.bank("train")[0]
        dev_man = self.bank("dev")[0]
        for name, man in (("train", train_man), ("dev", dev_man)):
            rep = banks.validate(man, (self.cfg["data"][name] or {}).get("expect"), check_audio=False)
            if not rep["ok"]:
                raise RuntimeError(f"{name} bank failed validation: {rep['failures']}")
        self.models.release()
        t0 = time.time()
        model, processor = self.loader()
        self.model_info = {"load_s": round(time.time() - t0, 1), "vram_after_load": vram()}
        quantized = self.cfg["model"]["quant"] in ("nf4", "fp4") and bool(hostmod.gpu_info().get("cuda"))
        model = lora_train.attach_lora(model, spec["recipe"], quantized=quantized)
        self.progress.start_items(s.name, total=0, done=0)

        def on_step(step, total):
            with self.progress.lock:
                st = self.progress.stages[s.name]
                st.update(done=step, total=total)
            if step == 1 or step % 10 == 0:
                self.log(f"[{s.name}] step {step}/{total}, VRAM {vram()}")

        report = lora_train.train(model, processor, train_man, dev_man, self.work / "lora_train", self.remote,
                                  f"checkpoints/{self.run_name}", recipe=spec["recipe"], log=self.log,
                                  on_step=on_step, max_steps=spec["max_steps"])
        lora = L.lora_report(model)
        if lora["n_lora_modules"] == 0 or lora["n_lora_B_zero"] == lora["n_lora_modules"]:
            raise RuntimeError(f"training left the LoRA untrained: {lora}")
        final = Path(report["final_dir"])
        for p in sorted(final.iterdir()):
            self.remote.put(p, f"{spec['output']}/{p.name}")
        report.update(lora_report=lora, adapter=lora_train.adapter_summary(final), output=spec["output"],
                      model_info=self.model_info, vram_peak=vram(), gpu=hostmod.gpu_info().get("devices"),
                      elapsed_s=round(time.time() - t0, 1))
        self.log(f"[{s.name}] adapter from {report['best_from']} saved to {spec['output']}, evals {report['evals']}")
        return report, {"retrain/report.json": json.dumps(report, indent=2, default=str)}

    def stage_data(self, s: Stage):
        man, sha = self.bank(s.which)
        d = self.cfg["data"][s.which]
        names = REPO_ROOT / d["names_csv"] if d.get("names_csv") else None
        rep = banks.validate(man, d.get("expect"), check_audio=True, names_csv=names)
        rep["sha256"] = sha
        if s.which == "test":
            rep["neutral_identical"] = self._neutral_identity(man)
        self.log(f"bank {s.which}: {json.dumps({k: rep[k] for k in ('rows', 'conditions', 'base_ids', 'rec_ids', 'speech_share', 'ok')})}")
        files = {f"data_{s.which}.json": json.dumps(rep, indent=2, default=str)}
        if not rep["ok"]:
            self.publish(files, f"{self.run_id}: data_{s.which} failed")
            raise RuntimeError(f"bank {s.which} failed validation: {rep['failures']}")
        return rep, files

    def _neutral_identity(self, man: pd.DataFrame):
        neutral = man[man["condition"].isin(NEUTRAL)]
        if neutral.empty:
            return None
        h = neutral["path"].map(lambda p: hashlib.sha256(L.load_wav(p).tobytes()).hexdigest())
        n = neutral.assign(h=h).groupby("base_id")["h"].nunique()
        return {"base_ids_with_identical_neutral_conditions": int((n == 1).sum()), "of": int(len(n))}

    def stage_check(self, s: Stage):
        b = self.models.get(s.backbone)
        model, proc, rec = b.model, b.processor, b.recorder
        wav = probe_wav()
        text = L.build_text(proc, L.PROMPTS[self.cfg["prompt_key"]])
        inp = L.prepare_inputs(proc, text, wav).to(L.model_device(model))
        s1, a1 = L.forward_collect(model, rec, inp, b.labels, PROBE_SEGMENT_MS, len(wav), b.audio_id, check_head=True)
        s2, a2 = L.forward_collect(model, rec, inp, b.labels, PROBE_SEGMENT_MS, len(wav), b.audio_id)
        tok = proc.tokenizer
        res = {"backbone": s.backbone, "probe_seconds": len(wav) / L.SR,
               "n_audio_tokens": s1["n_audio_tokens"], "n_audio_in_ids": s1.get("n_audio_in_ids"),
               "head_check_maxdiff": s1["head_check_maxdiff"], "head_tol": self.cfg["check"]["head_tol"],
               "zA": s1["zA"], "zB": s1["zB"], "s_primary": s1["s_primary"], "s_lse": s1["s_lse"],
               "deterministic": s1["s_primary"] == s2["s_primary"],
               "max_abs_dh_repeat": float(np.abs(a1["h_final"].astype(float) - a2["h_final"].astype(float)).max()),
               "top1": tok.decode([s1["top1"]]), "top1_is_label": s1["top1_is_label"],
               "labels": {**b.labels, "decoded": [tok.decode([i]) for i in b.labels["pos_ids"] + b.labels["neg_ids"]]},
               "feature_dims": {k: int(v.shape[0]) for k, v in a1.items()}, "recorder": rec.names,
               "n_llm_layers": rec.n_llm_layers, "model_info": self.model_info, "vram": vram(),
               "gpu": hostmod.gpu_info().get("devices")}
        failures = []
        if not np.isfinite([res["zA"], res["zB"], res["max_abs_dh_repeat"]]).all():
            failures.append(f"non-finite logits or states (zA {res['zA']}, zB {res['zB']}), check the fp16 path")
        if res["n_audio_in_ids"] != res["n_audio_tokens"]:
            failures.append(f"audio tokens in the text {res['n_audio_in_ids']} != {res['n_audio_tokens']}")
        if res["n_audio_tokens"] != 50:
            failures.append(f"a 2 s WAV gave {res['n_audio_tokens']} audio tokens, expected 50")
        if not res["head_check_maxdiff"] < res["head_tol"]:
            failures.append(f"hooked state does not reproduce the logits, max diff {res['head_check_maxdiff']:.3g}")
        if s.backbone == "lora":
            rep = b.lora_rep
            res["lora_report"] = rep
            if rep["n_lora_B_zero"]:
                failures.append(f"{rep['n_lora_B_zero']} LoRA modules still hold zero B matrices")
            if not rep["by_component"].get("audio_tower"):
                self.warn("the adapter has no LoRA modules in the audio encoder. The paper says it does. "
                          "Check the adapter before using its results")
            base = self.remote.get_json(self.r_state("check_base", "done"))
            if base and "s_primary" in (base.get("result") or {}):
                res["base_s_primary"] = base["result"]["s_primary"]
                res["differs_from_base"] = abs(res["base_s_primary"] - res["s_primary"]) > 1e-6
                if not res["differs_from_base"]:
                    failures.append("LoRA and base give the same score on the probe")
            else:
                self.warn("check_base is not done, so base and LoRA outputs were not compared")
        res["failures"] = failures
        self.log(f"check {s.backbone}: tokens {res['n_audio_tokens']}, head diff {res['head_check_maxdiff']:.2e}, "
                 f"s = {res['s_primary']:.4f}, top1 {res['top1']!r}, VRAM {res['vram']}")
        files = {f"checks_{s.backbone}.json": json.dumps(res, indent=2, default=str)}
        if failures:
            self.publish(files, f"{self.run_id}: check_{s.backbone} failed")
            raise CheckFailed(f"check_{s.backbone}: {failures}")
        return res, files

    def stage_extract(self, s: Stage):
        if s.which == "eval":
            man = pd.concat([self.eval_manifest(), self.content_free()], ignore_index=True)
        else:
            man = self.bank("train")[0]
        if self.cfg.get("audio_transform"):
            t0 = time.time()
            man = lib_v3.transform_audio(man, self.work / "transformed" / s.which, self.cfg["audio_transform"])
            self.log(f"[{s.name}] {len(man)} containers through {self.cfg['audio_transform']} "
                     f"in {time.time() - t0:.0f} s")
        b = self.models.get(s.backbone)
        rdir, ldir = self.r_store(s.backbone, s.which), self.l_store(s.backbone, s.which)
        rep = sync_store(self.remote, rdir, ldir, fetch_npz=False, log=self.log, warn=self.warn)
        ids = set(man["variant_id"].astype(str))
        ex = self.cfg["extraction"]
        store = SyncedFeatureStore(ldir, self.remote, rdir, shard_size=ex["shard_size"],
                                   on_add=lambda: self.progress.item(s.name),
                                   tags={"session_id": self.session_id, "host": self.host.name})
        already = store.done_ids()
        self.progress.start_items(s.name, total=len(ids), done=len(already & ids))
        if already - ids:
            self.warn(f"[{s.name}] {len(already - ids)} stored items are not in the manifest")
        if self.cfg.get("generate"):  # the answer the model writes, parsed like the study (genread.py)
            from . import genread
            genread.run_generation(b.model, b.processor, man, store, self.cfg["prompt_key"], b.labels,
                                   audio_token_id=b.audio_id, max_new_tokens=self.cfg["generate"]["max_new_tokens"],
                                   log_every=ex["log_every"], log=self.log)
        else:
            L.run_extraction(b.model, b.processor, b.recorder, man, store, self.cfg["prompt_key"], b.labels,
                             audio_token_id=b.audio_id, n_head_checks=ex["n_head_checks"],
                             log_every=ex["log_every"])
        missing = ids - self._store_ids(ldir)
        if missing:
            raise RuntimeError(f"[{s.name}] {len(missing)} items were not extracted, e.g. {sorted(missing)[:3]}")
        st = self.progress.snapshot()["stages"].get(s.name, {})
        res = {"n_items": len(ids), "ids_hash": banks.ids_hash(ids), "resumed_with": len(already & ids),
               "n_shards": len(list(ldir.glob("shard_*.csv"))), "s_per_item": st.get("s_per_item"),
               "resume": rep, "gpu": hostmod.gpu_info().get("devices"), "vram": vram()}
        return res, {f"extract_{s.backbone}_{s.which}.json": json.dumps(res, indent=2, default=str)}

    def _store_ids(self, ldir: Path) -> set:
        ids = set()
        for p in ldir.glob("shard_*.csv"):
            ids |= set(pd.read_csv(p, usecols=["variant_id"])["variant_id"].astype(str))
        return ids

    def _verify_extract(self, s: Stage, marker: dict) -> bool:
        """True when the committed shards on the remote hold exactly the recorded items."""
        ldir = self.l_store(s.backbone, s.which)
        sync_store(self.remote, self.r_store(s.backbone, s.which), ldir, fetch_npz=False,
                   log=self.log, warn=self.warn)
        return banks.ids_hash(self._store_ids(ldir)) == (marker.get("result") or {}).get("ids_hash")

    def _load_extracted(self, b: str, which: str):
        U, name = self.universe(), f"extract_{b}_{which}"
        ldir = self.l_store(b, which)
        for attempt in range(2):
            marker = self.remote.get_json(self.r_state(name, "done"))
            if marker is None or marker.get("config_hash") != self.spec_hash(U[name]):
                raise RuntimeError(f"{name} is not done with the current configuration")
            sync_store(self.remote, self.r_store(b, which), ldir, fetch_npz=True, log=self.log, warn=self.warn)
            meta, arrays = L.load_store(ldir) if any(ldir.glob("shard_*.csv")) else (None, None)
            if meta is not None and banks.ids_hash(meta["variant_id"]) == marker["result"]["ids_hash"]:
                return meta, arrays
            if attempt == 0:
                self.warn(f"{name}: the stored items differ from the extraction record. Extracting the missing ones")
                current = self.progress.current
                self.remote.delete(self.r_state(name, "done"))
                self.execute(U[name])
                self.progress.current = current
        raise RuntimeError(f"{name}: the stored items still differ from the extraction record")

    # ------------------------------------------------------------ Qwen3-Omni, second model (exploratory)
    def _load_omni(self):
        from . import omni
        return omni.load(self.cfg, self.host.secret("HF_TOKEN", required=False), work=self.work, log=self.log)

    def stage_omni(self, s: Stage):
        """Frozen readouts of Qwen3-Omni on the selected conditions, one store per track, and the
        registered reading of the pilot (docs/exploratory_log.md, "Qwen3-Omni pilot (29 October)").
        A track is a prompt, optionally on the audio rolled by shift_ms and on fewer conditions."""
        from . import omni
        o, ex = self.cfg["omni"], self.cfg["extraction"]
        tracks = omni.tracks(o)
        man = self.eval_manifest()
        man = man[man["condition"].isin(o["conditions"])].reset_index(drop=True)
        lacking = sorted(set(o["conditions"]) - set(man["condition"]))
        if lacking:
            raise RuntimeError(f"[{s.name}] the selected items lack conditions {lacking}")
        mans = {t["name"]: man[man["condition"].isin(t["conditions"] or o["conditions"])].reset_index(drop=True)
                for t in tracks}
        ids = {k: set(m["variant_id"].astype(str)) for k, m in mans.items()}
        stores, missing = {}, {}
        for t in tracks:
            k = t["name"]
            rdir, ldir = self.r_store("omni", k), self.l_store("omni", k)
            sync_store(self.remote, rdir, ldir, fetch_npz=False, log=self.log, warn=self.warn)
            stores[k] = SyncedFeatureStore(ldir, self.remote, rdir, shard_size=ex["shard_size"],
                                           on_add=lambda: self.progress.item(s.name),
                                           tags={"session_id": self.session_id, "host": self.host.name})
            missing[k] = len(ids[k] - stores[k].done_ids())
        total = sum(len(v) for v in ids.values())
        self.progress.start_items(s.name, total=total, done=total - sum(missing.values()))
        timing, checks = {}, None
        if any(missing.values()):  # a resumed run whose stores are complete does not load the model
            t0 = time.time()
            model, proc, info = self.omni_loader()
            info = {**info, "vram_after_load": vram(), "gpu": hostmod.gpu_info().get("devices")}
            self.log(f"[{s.name}] Qwen3-Omni loaded in {time.time() - t0:.0f} s: {json.dumps(info, default=str)}")
            checks = omni.check(model, proc, head_tol=self.cfg["check"]["head_tol"])
            checks["load"] = info
            self.log(f"[{s.name}] check: tokens {checks['n_audio_in_ids']}, head diff "
                     f"{checks['head_check_maxdiff']:.2e}, s = {checks['s_primary']:.4f}, "
                     f"top1 {checks['top1_decoded']!r}")
            self.publish({"checks_omni.json": json.dumps(checks, indent=2, default=str)},
                         f"{self.run_id}: Qwen3-Omni checks")
            if checks["failures"]:
                raise CheckFailed(f"{s.name}: {checks['failures']}")
            for t in tracks:
                if missing[t["name"]]:
                    timing[t["name"]] = omni.run(model, proc, mans[t["name"]], stores[t["name"]], t,
                                                 log_every=ex.get("log_every", 50), log=self.log)
            del model
        metas = {}
        for t in tracks:
            k = t["name"]
            csvs = sorted(self.l_store("omni", k).glob("shard_*.csv"))
            meta = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True) if csvs else pd.DataFrame()
            if len(meta):
                meta = meta.drop_duplicates("variant_id", keep="last")
                meta = meta[meta["variant_id"].astype(str).isin(ids[k])].reset_index(drop=True)
            if len(meta) != len(ids[k]):
                raise RuntimeError(f"[{s.name}] {k}: {len(ids[k]) - len(meta)} items were not scored")
            metas[k] = meta
        a = self.cfg["analysis"]
        table = omni.condition_table(metas, B=a["bootstrap_B"], seed=a["seed"])
        rates = [t["s_per_item"] for t in timing.values() if t.get("s_per_item")]
        verdict = omni.verdict(table, float(np.mean(rates)) if rates else None, o["rule"])
        res = {"n_items": {k: len(v) for k, v in ids.items()}, "n_passes": total,
               "ids_hash": banks.ids_hash(set().union(*ids.values())), "timing": timing, "verdict": verdict,
               "checks": {k: v for k, v in (checks or {}).items() if k != "labels"}}
        local = self.work / "results_omni" / self.run_name
        local.mkdir(parents=True, exist_ok=True)
        (local / "condition_table.csv").write_text(table.to_csv(index=False))
        (local / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
        for p in sorted(local.iterdir()):
            self.remote.put(p, f"{self.r_results('omni')}/{p.name}")
        self.log(f"[{s.name}] verdict: {json.dumps(verdict, default=str)}")
        files = {"results/omni/condition_table.csv": table.to_csv(index=False),
                 "results/omni/verdict.json": json.dumps(verdict, indent=2, default=str),
                 "extract_omni.json": json.dumps(res, indent=2, default=str)}
        return res, files

    def stage_analyze(self, s: Stage):
        meta, arrays = self._load_extracted(s.backbone, "eval")
        train = self._load_extracted(s.backbone, "train") if "data_train" in self.universe() else None
        a = self.cfg["analysis"]
        out = analysis.per_backbone(meta, arrays, train=train, B=a["bootstrap_B"], seed=a["seed"],
                                    prevalence_reps=a["prevalence_reps"])
        local = self.work / "results" / self.run_name / s.backbone
        shutil.rmtree(local, ignore_errors=True)
        local.mkdir(parents=True)
        for name, df in out["tables"].items():
            df.to_csv(local / f"{name}.csv", index=False)
        summary = {**out["summary"], "backbone": s.backbone, "run_name": self.run_name,
                   "code_sha": self.code_sha, "fit_info": out["fit_info"]}
        export.write_scores(local / "scores.csv.gz", out["meta"], out["readouts"], info=summary)
        (local / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        for p in sorted(local.iterdir()):
            self.remote.put(p, f"{self.r_results(s.backbone)}/{p.name}")
        cc = out["tables"]["contrasts"]
        self.log(f"analysis {s.backbone}:\n{cc.round(4).to_string(index=False)}")
        return out["summary"], {f"results/{s.backbone}/{p.name}": p for p in sorted(local.iterdir())}

    def stage_joint(self, s: Stage):
        U, runs = self.universe(), {}
        for b in BACKBONES:
            marker = self.remote.get_json(self.r_state(f"analyze_{b}", "done"))
            if marker is None or marker.get("config_hash") != self.spec_hash(U[f"analyze_{b}"]):
                raise RuntimeError(f"analyze_{b} is not done with the current configuration")
            local = self.work / "results" / self.run_name / b
            for name in ("scores.csv.gz", "scores.json"):
                self.remote.get(f"{self.r_results(b)}/{name}", local / name)
            runs[b] = export.load_scores(local / "scores.csv.gz")
        a = self.cfg["analysis"]
        out = analysis.joint(runs, B=a["bootstrap_B"], seed=a["seed"], h3_margin=a["h3_margin"])
        local = self.work / "results" / self.run_name / "joint"
        shutil.rmtree(local, ignore_errors=True)
        local.mkdir(parents=True)
        for name, df in out["tables"].items():
            df.to_csv(local / f"{name}.csv", index=False)
        summary = {**out["summary"], "run_name": self.run_name, "code_sha": self.code_sha}
        (local / "joint_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        for p in sorted(local.iterdir()):
            self.remote.put(p, f"work/{self.run_name}/joint/{p.name}")
        self.log(f"joint analysis:\n{out['tables']['joint_contrasts'].round(4).to_string(index=False)}")
        return summary, {f"results/joint/{p.name}": p for p in sorted(local.iterdir())}

    # ------------------------------------------------------------ session
    def env_json(self, plan) -> dict:
        m = self.cfg["model"]
        return {"run_id": self.run_id, "run_name": self.run_name, "session_id": self.session_id,
                "host": self.host.name, "started_at": _now(), "code_sha": self.code_sha,
                "code_ref": os.environ.get("ERCLM_CODE_REF"), "config_file": self.config_file,
                "config_hash": config_hash(self.cfg), "config": self.cfg,
                "model": {"id": m["id"], "revision": m["revision"], "quant": m["quant"]},
                "backbones": self.backbones, "stages": [s.name for s in plan], "force": sorted(self.force),
                "remote": self.remote.describe(), "work": str(self.work),
                "gpu": hostmod.gpu_info(), "versions": hostmod.package_versions()}

    def _error_md(self, exc: BaseException, stage: Optional[str]) -> str:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        tail = self.sink.tail_text() if self.sink else ""
        text = (f"# Error in {self.run_id}\n\n"
                f"- Session {self.session_id} on {self.host.name}\n"
                f"- Stage {stage or 'before the stages'}\n"
                f"- Time {_now()}\n"
                f"- Code {self.code_sha}\n\n"
                f"## Traceback\n\n```\n{tb}```\n\n"
                f"## Environment\n\n```json\n{json.dumps(self.env, indent=2, default=str)}\n```\n\n"
                f"## Last 200 lines of the log\n\n```\n{tail}\n```\n")
        return self.sink.redact(text) if self.sink else text

    def run(self) -> dict:
        log_path = self.work / "logs" / f"{self.session_id}.log"
        self.sink = LogSink(log_path, self.host.secret_values())
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old_out, self.sink), _Tee(old_err, self.sink)
        hb, current = None, None
        try:
            self.log(f"session {self.session_id} on {self.host.name}, run {self.run_id}, "
                     f"code {self.code_sha[:12]}, remote {self.remote.describe()}")
            plan = self.plan()
            self.log("plan: " + ", ".join(s.name for s in plan))
            self.env = self.env_json(plan)
            for s in plan:
                self.progress.set(s.name, status="pending")
            env_text = json.dumps(self.env, indent=2, default=str)
            self.runs.publish({"env.json": env_text, f"sessions/{self.session_id}.json": env_text},
                              delete=["error.md"], message=f"{self.run_id}: session {self.session_id} started")
            self.publish_progress()
            hbc = self.cfg["heartbeat"]
            hb = Heartbeat(self, hbc["log_every_s"], hbc["progress_every_s"])
            hb.start()
            for s in plan:
                current = s.name
                self.execute(s)
            current = None
            self.progress.current = None
            self.log("all stages finished")
            self.publish_progress()
            return self.progress.snapshot()
        except (Exception, KeyboardInterrupt) as exc:
            if current:
                self.progress.set(current, status="failed")
            self.progress.warn(f"failed in {current}: {exc!r}")
            text = self._error_md(exc, current)
            self.log(f"FAILED in {current}: {exc!r}")
            try:
                self.remote.put_bytes(text.encode(), f"logs/{self.run_id}/{self.session_id}.error.md")
            except Exception as e2:
                self.log(f"could not store error.md on the remote: {e2!r}")
            self.publish({"error.md": text, f"errors/{self.session_id}.md": text,
                          "progress.json": json.dumps({"run_id": self.run_id, "session_id": self.session_id,
                                                       "updated_at": _now(), "error": repr(exc),
                                                       **self.progress.snapshot()}, indent=2, default=str)},
                         f"{self.run_id}: error in {current}")
            raise
        finally:
            if hb is not None:
                hb.stop()
            try:
                self.upload_log()
            except Exception as exc:
                self.log(f"could not upload the log: {exc!r}")
            sys.stdout, sys.stderr = old_out, old_err
            self.sink.close()


# ============================================================================
# CLI
# ============================================================================

def build_remote(args, cfg: dict, host: hostmod.Host) -> Remote:
    local = args.remote_dir or os.environ.get("ERCLM_REMOTE_DIR")
    if local:
        return LocalRemote(local)
    conf = host.write_rclone_conf(host.secret("RCLONE_CONF"))
    return RcloneRemote(cfg["remote"], config=str(conf))


def ensure_pip(pins, argv, host_name: str, execv=os.execv) -> bool:
    """Installs the packages a config pins on top of requirements.txt and restarts the runner.

    Only on Colab and Kaggle, once per session (ERCLM_PIP holds the hash of what was
    installed). The restart keeps any module from running with the versions imported
    before the install. The Qwen3-Omni pilot pins transformers 4.57 this way (omni.py).
    """
    if not pins or host_name == "local":
        return False
    h = config_hash(list(pins))
    if os.environ.get("ERCLM_PIP") == h:
        return False
    import subprocess
    print(f"installing the config pins {list(pins)}", flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pins], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:], file=sys.stderr)
        raise SystemExit("pip install of the config pins failed")
    os.environ["ERCLM_PIP"] = h
    sys.stdout.flush()
    sys.stderr.flush()
    execv(sys.executable, [sys.executable, "-u", "-m", "erclm.runner", *argv])
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m erclm.runner", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="configs/<smoke|pilot|full>.yaml")
    ap.add_argument("--backbones", help="comma list, e.g. base or lora (default from the config)")
    ap.add_argument("--stages", help="comma list of stages to run (default: all in order)")
    ap.add_argument("--force", action="append", default=[], help="redo a stage (repeatable, or 'all')")
    ap.add_argument("--remote-dir", help="local folder instead of Drive, for debugging")
    ap.add_argument("--no-runs", action="store_true", help="do not push to the runs branch")
    ap.add_argument("--work", help="local working folder (default depends on the host)")
    ap.add_argument("--plan", action="store_true", help="print the stage plan and exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    for k, v in (cfg.get("env") or {}).items():  # before torch is imported, e.g. the CUDA allocator settings
        os.environ.setdefault(k, str(v))
    host = hostmod.Host.current(args.work)
    if not args.plan:
        ensure_pip(cfg.get("pip"), list(sys.argv[1:] if argv is None else argv), host.name)
    backbones =[b.strip() for b in args.backbones.split(",")] if args.backbones else None
    only = [s.strip() for s in args.stages.split(",")] if args.stages else None
    rid = run_id_for(cfg, backbones)
    remote = build_remote(args, cfg, host)
    if args.no_runs:
        runs = NullRunsLog()
    else:
        runs = RunsLog(os.environ.get("ERCLM_REPO") or cfg["repo"], host.secret("GH_TOKEN"),
                       host.work / "runs_clone", rid, redact=secret_fragments(host.secret_values()))
    runner = Runner(cfg, host, remote, runs, backbones=backbones, only=only, force=args.force, run_id=rid,
                    config_file=args.config)
    if args.plan:
        for s in runner.plan():
            print(s.name)
        return 0
    try:
        runner.run()
        return 0
    except (Exception, KeyboardInterrupt):
        print(f"\nThe run failed. See runs/{rid}/error.md and logs/{rid}/ on Drive.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
