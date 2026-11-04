"""The runner end to end on CPU with the tiny Qwen2-Audio, a fake Drive and a fake GitHub.

Each session gets a fresh working folder, as after "Disconnect and delete
runtime". The four failure cases of docs/running.md are here: a
cut between the npz and the CSV upload, a corrupt npz on the remote, a
duplicated shard and a stage marked done with another configuration. So are a
cut in the middle of a shard and a cut during an upload.
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import helpers
from helpers import SimulatedCrash
from erclm import analysis, export
from erclm import lib_v2 as L
from erclm.gitlog import RunsLog
from erclm.host import Host
from erclm.runner import Runner, StageConfigMismatch, CheckFailed

N_EVAL = 6 * 2 * 7 + 3          # 12 pilot clips x 7 conditions + 3 content-free containers
N_TRAIN = 48


@pytest.fixture
def world(tmp_path):
    return helpers.build_world(tmp_path, train=False)


@pytest.fixture
def world_train(tmp_path):
    return helpers.build_world(tmp_path, train=True)


def session(world, cfg, n, remote=None, **kw):
    work = world["root"] / f"vm{n}"
    runs = RunsLog(str(world["github"]), None, work / "runs_clone", kw.pop("run_id", cfg["run_name"]), backoff_s=0)
    return Runner(cfg, Host("local", work), remote or world["remote"], runs, model_loader=helpers.tiny_loader,
                  session_id=f"s{n}", **kw)


def stored(world, cfg, b="base", which="eval"):
    d = world["remote"].root / "work" / cfg["run_name"] / b / which
    meta, arrays = L.load_store(d)
    return meta, arrays, sorted(p.name for p in d.iterdir())


def marker(world, cfg, stage, kind="done"):
    p = world["remote"].root / "state" / cfg["run_name"] / f"{stage}.{kind}.json"
    return json.loads(p.read_text()) if p.exists() else None


class CountingForward:
    """Counts forwards and optionally kills the VM at one of them."""

    def __init__(self, monkeypatch, crash_at=None):
        self.n, self.crash_at, self.orig = 0, crash_at, L.forward_collect
        monkeypatch.setattr(L, "forward_collect", self)

    def __call__(self, *a, **k):
        self.n += 1
        if self.crash_at is not None and self.n == self.crash_at:
            raise SimulatedCrash(f"VM died at forward {self.n}")
        return self.orig(*a, **k)


class CrashingRemote(helpers.LocalRemote):
    """Dies once, on the put of a given file, either before it starts or half way through."""

    def __init__(self, root, target, during=False):
        super().__init__(root)
        self.target, self.during, self.fired = target, during, False

    def put(self, local, rel):
        if not self.fired and rel.endswith(self.target) and not self.during:
            self.fired = True
            raise SimulatedCrash(f"VM died before uploading {rel}")
        return super().put(local, rel)

    def _upload(self, local, rel):
        if not self.fired and self.during and rel.endswith(self.target + ".part"):
            self.fired = True
            dst = self._p(rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            data = Path(local).read_bytes()
            dst.write_bytes(data[: len(data) // 2])
            raise SimulatedCrash(f"VM died while uploading {rel}")
        return super()._upload(local, rel)


def assert_complete(world, cfg, n_items=N_EVAL):
    meta, arrays, names = stored(world, cfg)
    assert len(meta) == n_items and meta["variant_id"].is_unique
    assert meta["session_id"].notna().all() and set(meta["host"]) == {"local"}
    assert all(len(v) == n_items for v in arrays.values())
    assert not [n for n in names if n.endswith(".part")]
    csvs = {n[:-4] for n in names if n.endswith(".csv")}
    assert csvs == {n[:-4] for n in names if n.endswith(".npz")} == {n[:-5] for n in names if n.endswith(".json")}
    return meta


# ---------------------------------------------------------------- full run

def test_full_run_publishes_and_scores_reproduce_the_contrasts(world_train, monkeypatch):
    w, cfg = world_train, helpers.tiny_config(train=True)
    snap = session(w, cfg, 1).run()
    stages = ["data_test", "data_train", "check_base", "extract_base_eval", "extract_base_train", "check_lora",
              "extract_lora_eval", "extract_lora_train", "analyze_base", "analyze_lora", "joint"]
    assert [k for k in snap["stages"]] == stages
    assert all(st["status"] == "done" for st in snap["stages"].values())
    for s in stages:
        assert marker(w, cfg, s)["published"] is True

    gh = w["github"]
    env = json.loads(helpers.show(gh, "pilot-test/env.json"))
    assert env["model"]["revision"] == "test" and env["session_id"] == "s1"
    assert helpers.show(gh, "pilot-test/error.md") is None
    data = json.loads(helpers.show(gh, "pilot-test/data_test.json"))
    assert data["ok"] and data["rows"] == 140 and data["neutral_identical"]["of"] == 20
    lora = json.loads(helpers.show(gh, "pilot-test/checks_lora.json"))
    assert lora["n_audio_tokens"] == 50 and lora["head_check_maxdiff"] < 1e-2
    assert lora["lora_report"]["n_lora_B_zero"] == 0 and lora["lora_report"]["by_component"]["audio_tower"] > 0
    assert lora["differs_from_base"] is True
    assert json.loads(helpers.show(gh, "pilot-test/extract_base_eval.json"))["n_items"] == N_EVAL
    assert json.loads(helpers.show(gh, "pilot-test/extract_lora_train.json"))["n_items"] == N_TRAIN
    assert helpers.show(gh, "pilot-test/progress.json") is not None
    assert w["remote"].exists("logs/pilot-test/s1.log")

    # A reader rebuilds the joint analysis from the published score tables alone
    tables = {}
    for b in ("base", "lora"):
        d = w["root"] / "reader" / b
        d.mkdir(parents=True)
        for name in ("scores.csv.gz", "scores.json"):
            (d / name).write_bytes(helpers.show(gh, f"pilot-test/results/{b}/{name}"))
        tables[b] = export.load_scores(d / "scores.csv.gz")
        assert len(tables[b][0]) == 84 and "L2" in tables[b][1]       # confirmatory heads
    out = analysis.joint(tables, B=30, seed=42, h3_margin=0.02)
    mine = out["tables"]["joint_contrasts"].reset_index(drop=True)
    theirs = pd.read_csv(io.BytesIO(helpers.show(gh, "pilot-test/results/joint/joint_contrasts.csv")))
    assert list(mine.contrast) == list(theirs.contrast)
    np.testing.assert_allclose(mine[["estimate", "lo", "hi", "se"]].to_numpy(float),
                               theirs[["estimate", "lo", "hi", "se"]].to_numpy(float), rtol=0, atol=1e-12)
    assert out["summary"]["confirmatory"] is True

    # per-backbone contrasts are reproducible from the score table too
    m, R = tables["base"]
    tab = L.condition_table(m, R)
    cc_ref = pd.read_csv(io.BytesIO(helpers.show(gh, "pilot-test/results/base/contrasts.csv")))
    _, cc = L.cluster_bootstrap(m, R, B=30, seed=42, metrics=("auroc", "ba", "oracle_ba"),
                                contrasts=analysis.backbone_contrasts(tab, "L2", "P_enc_region_mean"))
    np.testing.assert_allclose(cc["estimate"].to_numpy(float), cc_ref["estimate"].to_numpy(float), atol=1e-9)

    # a second session finds everything done and does no work
    fwd = CountingForward(monkeypatch)
    snap2 = session(w, cfg, 2).run()
    assert fwd.n == 0 and all(st.get("skipped") for st in snap2["stages"].values())


# ---------------------------------------------------------------- cuts

def test_cut_between_npz_and_csv_upload(world):
    cfg = helpers.tiny_config(train=False)
    only = ["extract_base_eval"]
    crashing = CrashingRemote(world["remote"].root, "base/eval/shard_00002.csv")
    with pytest.raises(SimulatedCrash):
        session(world, cfg, 1, remote=crashing, only=only).run()
    names = sorted(p.name for p in (world["remote"].root / "work/pilot-test/base/eval").iterdir())
    assert "shard_00002.npz" in names and "shard_00002.csv" not in names     # npz up, CSV not
    session(world, cfg, 2, only=only).run()
    assert_complete(world, cfg)


def test_cut_during_an_upload(world):
    cfg = helpers.tiny_config(train=False)
    only = ["extract_base_eval"]
    crashing = CrashingRemote(world["remote"].root, "base/eval/shard_00003.npz", during=True)
    with pytest.raises(SimulatedCrash):
        session(world, cfg, 1, remote=crashing, only=only).run()
    assert (world["remote"].root / "work/pilot-test/base/eval/shard_00003.npz.part").exists()
    session(world, cfg, 2, only=only).run()
    assert_complete(world, cfg)


def test_cut_in_the_middle_of_a_shard_loses_less_than_a_shard(world, monkeypatch):
    cfg = helpers.tiny_config(train=False)
    only = ["extract_base_eval"]
    fwd = CountingForward(monkeypatch, crash_at=19)       # shard size 7: two shards committed, 4 items lost
    with pytest.raises(SimulatedCrash):
        session(world, cfg, 1, only=only).run()
    fwd.crash_at, fwd.n = None, 0
    session(world, cfg, 2, only=only).run()
    assert fwd.n == N_EVAL - 14
    assert_complete(world, cfg)


@pytest.mark.skipif(not __import__("shutil").which("rclone"), reason="rclone not installed")
def test_cut_and_resume_through_rclone(world, monkeypatch):
    from erclm.storage import RcloneRemote
    cfg = helpers.tiny_config(train=False)
    root = world["remote"].root
    (root.parent / "empty.conf").write_text("")
    rc = RcloneRemote(str(root), config=str(root.parent / "empty.conf"))
    fwd = CountingForward(monkeypatch, crash_at=12)
    with pytest.raises(SimulatedCrash):
        session(world, cfg, 1, remote=rc, only=["extract_base_eval"]).run()
    fwd.crash_at = None
    session(world, cfg, 2, remote=rc, only=["extract_base_eval", "analyze_base"]).run()
    assert_complete(world, cfg)
    assert marker(world, cfg, "analyze_base")["result"]["n_items"] == N_EVAL - 3


# ---------------------------------------------------------------- remote damage

def test_corrupt_npz_on_the_remote_is_redone(world, monkeypatch):
    cfg = helpers.tiny_config(train=False)
    session(world, cfg, 1, only=["extract_base_eval"]).run()
    npz = world["remote"].root / "work/pilot-test/base/eval/shard_00001.npz"
    npz.write_bytes(npz.read_bytes()[:1000])                              # truncated on Drive
    fwd = CountingForward(monkeypatch)
    snap = session(world, cfg, 2, only=["extract_base_eval", "analyze_base"]).run()
    assert fwd.n == 7                                                      # only that shard again
    assert any("truncated or corrupt" in w for w in snap["warnings"])
    assert_complete(world, cfg)
    assert marker(world, cfg, "analyze_base")["result"]["n_items"] == N_EVAL - 3


def test_duplicated_shard_is_deduplicated(world):
    cfg = helpers.tiny_config(train=False)
    session(world, cfg, 1, only=["extract_base_eval"]).run()
    d = world["remote"].root / "work/pilot-test/base/eval"
    for ext in ("npz", "json", "csv"):                                     # same items committed twice
        (d / f"shard_00099.{ext}").write_bytes((d / f"shard_00000.{ext}").read_bytes())
    session(world, cfg, 2, only=["extract_base_eval", "analyze_base"]).run()
    meta, _, _ = stored(world, cfg)
    assert len(meta) == N_EVAL and meta["variant_id"].is_unique
    assert marker(world, cfg, "analyze_base")["result"]["n_items"] == N_EVAL - 3


# ---------------------------------------------------------------- configuration changes

def test_extraction_done_with_another_config_stops_until_forced(world):
    cfg = helpers.tiny_config(train=False)
    session(world, cfg, 1, only=["extract_base_eval"]).run()
    before = stored(world, cfg)[2]
    cfg2 = helpers.tiny_config(train=False, llm_layers=[0, 1])
    with pytest.raises(StageConfigMismatch, match="llm_layers"):
        session(world, cfg2, 2, only=["extract_base_eval"]).run()
    assert stored(world, cfg)[2] == before                                 # nothing touched
    err = helpers.show(world["github"], "pilot-test/error.md").decode()
    assert "StageConfigMismatch" in err and "--force extract_base_eval" in err
    assert helpers.show(world["github"], "pilot-test/errors/s2.md") is not None
    session(world, cfg2, 3, only=["extract_base_eval"], force=["extract_base_eval"]).run()
    meta, arrays, _ = stored(world, cfg2)
    assert "h_layer1" in arrays and "h_layer2" not in arrays and len(meta) == N_EVAL
    assert helpers.show(world["github"], "pilot-test/error.md") is None    # cleared by the new session


def test_extraction_started_with_another_config_stops(world, monkeypatch):
    cfg = helpers.tiny_config(train=False)
    CountingForward(monkeypatch, crash_at=10)
    with pytest.raises(SimulatedCrash):
        session(world, cfg, 1, only=["extract_base_eval"]).run()
    cfg2 = helpers.tiny_config(train=False, prompt_key="opro_tmpl")
    with pytest.raises(StageConfigMismatch, match="was started"):
        session(world, cfg2, 2, only=["extract_base_eval"]).run()


def test_analysis_done_with_another_config_is_redone_with_a_warning(world):
    cfg = helpers.tiny_config(train=False)
    session(world, cfg, 1, only=["extract_base_eval", "analyze_base"]).run()
    first = marker(world, cfg, "analyze_base")
    cfg2 = helpers.tiny_config(train=False, analysis={"bootstrap_B": 40})
    snap = session(world, cfg2, 2, only=["extract_base_eval", "analyze_base"]).run()
    second = marker(world, cfg2, "analyze_base")
    assert second["config_hash"] != first["config_hash"] and second["session_id"] == "s2"
    assert any("analyze_base" in w and "Redoing" in w for w in snap["warnings"])
    assert snap["stages"]["extract_base_eval"].get("skipped")


# ---------------------------------------------------------------- checks and session plumbing

def test_failed_check_publishes_its_report(world):
    cfg = helpers.tiny_config(train=False, check={"head_tol": 0.0})
    with pytest.raises(CheckFailed):
        session(world, cfg, 1, only=["check_base"]).run()
    rep = json.loads(helpers.show(world["github"], "pilot-test/checks_base.json"))
    assert rep["failures"] and "logits" in rep["failures"][0]
    assert marker(world, cfg, "check_base") is None


def test_smoke_and_heartbeat(world):
    cfg = helpers.tiny_config(train=False, heartbeat={"log_every_s": 0.05, "progress_every_s": 0.05})
    puts = []

    class Watching(helpers.LocalRemote):
        def put(self, local, rel):
            puts.append(rel)
            return super().put(local, rel)

    remote = Watching(world["remote"].root)
    session(world, cfg, 1, remote=remote, only=["smoke", "check_base"]).run()
    assert json.loads(helpers.show(world["github"], "pilot-test/smoke.json"))["bytes"] == 8 << 20
    assert not world["remote"].ls("state/smoke")
    assert puts.count("logs/pilot-test/s1.log") >= 2                      # heartbeat plus the final upload


def test_single_backbone_session_has_its_own_run_id(world):
    cfg = helpers.tiny_config(train=False)
    r = session(world, cfg, 1, backbones=["base"], run_id="pilot-test-base")
    assert [s.name for s in r.plan()] == ["data_test", "check_base", "extract_base_eval", "analyze_base"]


def test_prompt_variant_and_axis_selection(world):
    """Closing plan, decision D1: an exploratory prompt registered by lib_v3 and an axis filter."""
    cfg = helpers.tiny_config(train=False, run_name="prompt-test", prompt_key="ba_speech_first",
                              selection={"axes": ["duration"]},
                              stages=["data_test", "check_base", "extract_base_eval"])
    snap = session(world, cfg, 1, backbones=["base"]).run()
    assert [k for k in snap["stages"]] == ["data_test", "check_base", "extract_base_eval"]
    assert all(st["status"] == "done" for st in snap["stages"].values())
    meta, _, _ = stored(world, cfg)
    assert set(meta["prompt"]) == {"ba_speech_first"}
    items = meta[meta["role"] != "content_free"]
    assert set(items["axis"]) == {"duration"} and len(items) == 12 * 3
    assert L.PROMPTS["ab"]["labels"] == ("A", "B")


def test_provenance_stage_runs_only_when_asked(world):
    cfg = helpers.tiny_config(train=False, provenance={"name": "prov_bank_test", "exclude_clips": "x.csv",
                                                       "sources": {}})
    r = session(world, cfg, 1, backbones=["base"])
    assert "build_prov" in r.universe()
    assert "build_prov" not in [s.name for s in r.plan()]


def test_audio_transform_reaches_the_model_and_the_spec(world):
    """GPU plan of 22 Oct, I1: every container the model hears goes through the band-pass."""
    spec = {"kind": "bandpass", "lo": 60, "hi": 6000, "order": 8}
    cfg = helpers.tiny_config(train=False, run_name="cue-test", audio_transform=spec,
                              stages=["data_test", "check_base", "extract_base_eval"])
    r = session(world, cfg, 1, backbones=["base"])
    snap = r.run()
    assert all(st["status"] == "done" for st in snap["stages"].values())
    assert marker(world, cfg, "extract_base_eval")["config"]["audio_transform"] == spec
    assert len(list((r.work / "transformed" / "eval").glob("*.wav"))) == N_EVAL
    plain = helpers.tiny_config(train=False, run_name="plain-test", stages=["data_test", "check_base", "extract_base_eval"])
    session(world, plain, 2, backbones=["base"]).run()
    a, b = stored(world, cfg)[0].set_index("variant_id"), stored(world, plain)[0].set_index("variant_id")
    assert "audio_transform" not in marker(world, plain, "extract_base_eval")["config"]
    assert not np.allclose(a["s_primary"], b.loc[a.index, "s_primary"])


def test_lora_keep_restricts_the_adapter(world):
    """GPU plan of 22 Oct, I2: the LoRA kept in the language model only."""
    cfg = helpers.tiny_config(train=False, run_name="split-test", backbones=["lora"], lora_keep="language_model",
                              stages=["data_test", "check_lora", "extract_lora_eval"])
    snap = session(world, cfg, 1).run()
    assert all(st["status"] == "done" for st in snap["stages"].values())
    rep = json.loads(helpers.show(world["github"], "split-test/checks_lora.json"))["lora_report"]
    assert rep["restricted"]["kept"] == {"language_model": 12}
    assert rep["restricted"]["disabled"] == {"audio_tower": 6}
    assert marker(world, cfg, "extract_lora_eval")["config"]["lora_keep"] == "language_model"
    assert marker(world, cfg, "check_lora")["config"]["lora_keep"] == "language_model"
