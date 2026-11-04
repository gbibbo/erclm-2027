"""LoRA retraining (recipe of the study under review) on CPU with the tiny Qwen2-Audio, a fake Drive and a cut mid-training."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import helpers
from helpers import PROC, SimulatedCrash, make_model
from erclm import banks
from erclm import lib_v2 as L
from erclm import lora_train as T
from erclm.storage import LocalRemote

FAST = {**T.RECIPE, "batch_size": 2, "grad_accum": 1, "lr": 1e-3, "warmup_steps": 0, "checkpoint_every": 2,
        "eval_steps": 4, "lora_r": 4, "lora_alpha": 8}


def write_dev_bank(d: Path, n=4) -> Path:
    """A few factorial variants of recordings that are not in the train bank."""
    audio = d / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    rows = []
    for lab in (1, 0):
        for r in range(n):
            stem = f"voxconverse_dev{r:02d}z_0001" if lab else f"esc50_4-{4000 + r}-A-{r}_0001"
            name = f"{stem}_1000ms_dur1000ms_snr+20dB.wav"
            x = rng.standard_normal(32000).astype(np.float32) * (0.2 if lab else 0.05)
            sf.write(str(audio / name), x, 16000)
            rows.append({"audio_path": name, "ground_truth": "SPEECH" if lab else "NONSPEECH"})
    pd.DataFrame(rows).to_csv(d / "dev_metadata.csv", index=False)
    return d / "dev_metadata.csv"


def manifests(root: Path):
    tr = helpers.write_train_bank(root / "train")
    dv = write_dev_bank(root / "dev")
    return (L.load_manifest(tr, audio_root=root / "train" / "audio", role="train"),
            L.load_manifest(dv, audio_root=root / "dev" / "audio", role="train"))


def test_dataset_masks_everything_before_the_answer(tmp_path):
    train, _ = manifests(tmp_path)
    ds = T.AnswerDataset(train, PROC)
    for i in (0, len(train) - 1):
        item = ds[i]
        lab = item["labels"]
        first = int((lab != -100).nonzero()[0])
        assert (lab[:first] == -100).all() and first > len(lab) - 6
        expected = "A" if int(train.iloc[i]["label"]) == 1 else "B"
        assert PROC.tokenizer.decode([int(item["input_ids"][first])]).strip() == expected
    batch = T.collate([ds[0], ds[1], ds[2]])
    assert batch["input_features"].shape[0] == 3 and batch["labels"].shape == batch["input_ids"].shape


def test_train_resumes_after_a_cut_and_keeps_the_best_eval(tmp_path):
    train, dev = manifests(tmp_path)
    remote = LocalRemote(tmp_path / "drive")

    def crash_at_3(step, total):
        if step == 3:
            raise SimulatedCrash("VM died at step 3")

    model = T.attach_lora(make_model(PROC, seed=0), FAST, quantized=False)
    with pytest.raises(SimulatedCrash):
        T.train(model, PROC, train, dev, tmp_path / "vm1", remote, "checkpoints/t", recipe=FAST,
                log=lambda m: None, on_step=crash_at_3, max_steps=6)
    assert T.latest_committed(remote, "checkpoints/t") == "checkpoints/t/checkpoint-2"
    model = T.attach_lora(make_model(PROC, seed=0), FAST, quantized=False)
    rep = T.train(model, PROC, train, dev, tmp_path / "vm2", remote, "checkpoints/t", recipe=FAST,
                  log=lambda m: None, max_steps=6)
    assert rep["resumed_from"] == "checkpoints/t/checkpoint-2" and rep["global_step"] == 6
    assert [e["step"] for e in rep["evals"]] == [4, 6]                   # every eval_steps and at the end
    best = min(rep["evals"], key=lambda e: e["eval_loss"])["step"]
    assert rep["best_from"] == f"checkpoint-{best}"
    assert remote.exists("checkpoints/t/checkpoint-4/COMMIT.json")       # eval step kept
    assert remote.exists("checkpoints/t/checkpoint-6/COMMIT.json")       # last checkpoint kept
    assert not remote.exists("checkpoints/t/checkpoint-2/COMMIT.json")   # older one pruned
    loaded, lrep = L.load_lora(make_model(PROC, seed=0), rep["final_dir"])
    assert lrep["n_lora_B_zero"] == 0 and lrep["by_component"].get("audio_tower", 0) > 0


def test_runner_retrains_and_the_pilot_checks_the_new_adapter(tmp_path):
    from erclm.gitlog import RunsLog
    from erclm.host import Host
    from erclm.runner import DEFAULTS, Runner, _merge
    remote = LocalRemote(tmp_path / "drive")
    gh = helpers.make_github(tmp_path)
    src = tmp_path / "src"
    for name, csv, audio in (("train_bank_v2", helpers.write_train_bank(src / "train"), src / "train" / "audio"),
                             ("dev_bank_v2", write_dev_bank(src / "dev"), src / "dev" / "audio")):
        tar, _ = banks.make_tar(name, csv, audio, tmp_path / "packed")
        remote.put(tar, f"data/{name}.tar")
        remote.put(tar.with_name(tar.name + ".sha256"), f"data/{name}.tar.sha256")
    data = {"train": {"tar": "data/train_bank_v2.tar", "manifest": "train_bank_v2/manifest.csv",
                      "audio": "train_bank_v2/audio", "expect": {"rows": 48, "speech_share": 0.5}},
            "dev": {"tar": "data/dev_bank_v2.tar", "manifest": "dev_bank_v2/manifest.csv",
                    "audio": "dev_bank_v2/audio", "expect": {"rows": 8}}}
    cfg = _merge(DEFAULTS, {"run_name": "retrain-test", "model": {"revision": "x"}, "llm_layers": [0, 2],
                            "stages": ["train_lora"], "data": data,
                            "retrain": {"output": "models/lora_test", "recipe": FAST, "max_steps": 4}})
    runs = RunsLog(str(gh), None, tmp_path / "vm1" / "runs_clone", "retrain-test", backoff_s=0)
    Runner(cfg, Host("local", tmp_path / "vm1"), remote, runs, model_loader=helpers.tiny_loader,
           session_id="r1").run()
    assert set(remote.ls("models/lora_test")) >= {"adapter_config.json", "adapter_model.safetensors"}
    rep = json.loads(helpers.show(gh, "retrain-test/retrain/report.json"))
    assert rep["best_from"].startswith("checkpoint-") and rep["lora_report"]["n_lora_B_zero"] == 0
    pcfg = _merge(DEFAULTS, {"run_name": "pilot-test", "model": {"revision": "x"}, "llm_layers": [0, 2],
                             "adapter": "models/lora_test"})
    runs2 = RunsLog(str(gh), None, tmp_path / "vm2" / "runs_clone", "pilot-test", backoff_s=0)
    snap = Runner(pcfg, Host("local", tmp_path / "vm2"), remote, runs2, model_loader=helpers.tiny_loader,
                  session_id="p1", only=["check_base", "check_lora"]).run()
    assert snap["stages"]["check_lora"]["status"] == "done"
    checks = json.loads(helpers.show(gh, "pilot-test/checks_lora.json"))
    assert checks["lora_report"]["n_lora_B_zero"] == 0 and checks["differs_from_base"] is True
