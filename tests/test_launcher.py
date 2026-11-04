"""The committed launcher matches its builder, has three code cells and no experiment logic."""
import importlib.util
import json
from pathlib import Path

import nbformat
import pytest

import helpers

ROOT = helpers.HERE.parent


def _builder():
    spec = importlib.util.spec_from_file_location("build_launcher", ROOT / "tools" / "build_launcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launcher_matches_builder_and_compiles(tmp_path):
    B = _builder()
    fresh = nbformat.read(str(B.build(tmp_path / "l.ipynb")), as_version=4)
    committed = nbformat.read(str(ROOT / "notebooks" / "launcher.ipynb"), as_version=4)
    nbformat.validate(committed)
    assert [(c.cell_type, c.source) for c in committed.cells] == [(c.cell_type, c.source) for c in fresh.cells], \
        "run python tools/build_launcher.py"
    code = [c for c in committed.cells if c.cell_type == "code"]
    assert len(code) == 3
    for c in code:
        compile(c.source, f"<{c.id}>", "exec")
        assert "lib_v2" not in c.source and "drive.mount" not in c.source
    assert "erclm.runner" in code[2].source and 'CODE_REF = "main"' in code[0].source


def test_kaggle_is_detected_before_colab(monkeypatch):
    """Kaggle's image also sets COLAB_RELEASE_TAG (seen on 2026-10-16)."""
    from erclm import host
    monkeypatch.delenv("ERCLM_HOST", raising=False)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab")
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert host.detect() == "kaggle"
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE")
    if not __import__("os").path.isdir("/kaggle/working"):
        assert host.detect() == "colab"
    cell1 = nbformat.read(str(ROOT / "notebooks" / "launcher.ipynb"), as_version=4).cells[1].source
    assert cell1.index('"kaggle" if') < cell1.index('"colab" if')


def test_kaggle_kernel_folder(tmp_path):
    spec = importlib.util.spec_from_file_location("kaggle_push", ROOT / "tools" / "kaggle_push.py")
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    out = K.build_kernel(tmp_path / "k", "smoke", extra="--force check_base")
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["id"] == "gabrielbibb/erclm-launcher" and meta["enable_gpu"] and meta["enable_internet"]
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["is_private"] is True
    assert meta["dataset_sources"] == ["gabrielbibb/erclm-credentials"]
    cell1 = nbformat.read(str(out / "launcher.ipynb"), as_version=4).cells[1].source
    assert "CONFIG = 'smoke'" in cell1 and "EXTRA_ARGS = '--force check_base'" in cell1
    assert "CODE_REF = 'main'" in cell1
    cpu = json.loads((K.build_kernel(tmp_path / "c", "smoke", gpu=False) / "kernel-metadata.json").read_text())
    assert cpu["enable_gpu"] is False and cpu["machine_shape"] == ""



def test_configs_load_and_plan(tmp_path):
    from erclm.gitlog import NullRunsLog
    from erclm.host import Host
    from erclm.runner import Runner, load_config
    from erclm.storage import LocalRemote
    plans = {}
    for name in ("smoke", "pilot", "full"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg["model"]["revision"] == "0a095220c30b7b31434169c3086508ef3ea5bf0a"
        assert cfg["model"]["quant"] == "nf4" and cfg["prompt_key"] == "ab"
        r = Runner(cfg, Host("local", tmp_path / name), LocalRemote(tmp_path / "drive"), NullRunsLog())
        plans[name] = [s.name for s in r.plan()]
    assert plans["smoke"] == ["smoke", "check_base"]
    assert plans["pilot"] == ["data_test", "check_base", "extract_base_eval", "check_lora", "extract_lora_eval",
                              "analyze_base", "analyze_lora", "joint"]
    assert "extract_lora_train" in plans["full"] and plans["full"][-1] == "joint"
    assert json.dumps(load_config(ROOT / "configs" / "full.yaml")["analysis"]["bootstrap_B"]) == "10000"


def test_rclone_conf_plain_or_base64(tmp_path, monkeypatch):
    import base64
    from erclm.host import Host, decode_rclone_conf
    conf = '[erclm]\ntype = drive\nscope = drive.file\ntoken = {"access_token":"ya29.SECRETSECRET","refresh_token":"1//REFRESHREFRESH"}\n'
    b64 = base64.b64encode(conf.encode()).decode()
    assert decode_rclone_conf(conf) == conf
    assert decode_rclone_conf(b64) == conf
    assert decode_rclone_conf(base64.b64encode(b"\xef\xbb\xbf" + conf.encode()).decode()) == conf  # BOM from Windows
    assert decode_rclone_conf("not-base64-at-all!") == "not-base64-at-all!"
    h = Host("local", tmp_path)
    assert h.write_rclone_conf(b64).read_text() == conf
    monkeypatch.setenv("RCLONE_CONF", b64)
    from erclm.runner import secret_fragments
    frags = secret_fragments(h.secret_values())
    assert "1//REFRESHREFRESH" in frags and "ya29.SECRETSECRET" in frags

