"""Bank generator v2 on synthetic VoxConverse-like and ESC-50-like sources (no network)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import helpers
from erclm import bankgen as G
from erclm import banks
from erclm import lib_v2 as L

COUNTS = {"test": {"speech": 6, "nonspeech": 6, "speech_recordings": 3},
          "train": {"speech": 2, "nonspeech": 2}, "dev": {"speech": 2, "nonspeech": 2}}


def synth_sources(root: Path, n_vox: int = 8, n_esc: int = 20):
    rng = np.random.default_rng(0)
    vox = {}
    (root / "vox").mkdir(parents=True)
    for i in range(n_vox):
        rec = f"rec{chr(97 + i)}x"
        t = np.arange(20 * 16000) / 16000
        x = rng.standard_normal(len(t)) * 0.002
        segs = [(1.0, 6.0), (9.0, 8.0)]
        for s, d in segs:
            m = (t >= s) & (t < s + d)
            x[m] += 0.3 * np.sin(2 * np.pi * (140 + 10 * i) * t[m]) * (1 + np.sin(2 * np.pi * 4 * t[m]))
        sf.write(str(root / "vox" / f"{rec}.wav"), x.astype(np.float32), 16000, subtype="PCM_16")
        (root / "vox" / f"{rec}.rttm").write_text(
            "".join(f"SPEAKER {rec} 1 {s:.3f} {d:.3f} <NA> <NA> spk00 <NA> <NA>\n" for s, d in segs))
        vox[rec] = {"wav": root / "vox" / f"{rec}.wav", "rttm": root / "vox" / f"{rec}.rttm"}
    rows = []
    (root / "esc").mkdir()
    for j in range(n_esc):
        fold, src, take, target = 1 + j % 5, 1000 + j, "A", j % 7
        name = f"{fold}-{src}-{take}-{target}.wav"
        x = np.zeros(5 * 44100)
        a = 44100 * (1 + j % 3)
        x[a:a + 44100] = rng.standard_normal(44100) * 0.2
        sf.write(str(root / "esc" / name), x.astype(np.float32), 44100, subtype="PCM_16")
        rows.append({"filename": name, "fold": fold, "target": target, "category": f"c{target}",
                     "esc10": False, "src_file": src, "take": take, "path": str(root / "esc" / name)})
    esc = pd.DataFrame(rows)
    return vox, esc, {r["filename"]: Path(r["path"]) for r in rows}


def tiny_rooms():
    rng = np.random.default_rng(3)
    out = {}
    for t60 in G.T60S:
        n = int(t60 * 16000)
        rir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
        rir[0] = 3.0
        out[t60] = [{"rir": rir / np.sqrt(np.sum(rir ** 2)), "rt60": t60}]
    return out


def test_condition_names_parse_with_lib_v2():
    conds = G.test_conditions()
    assert len(conds) == 22 and len(set(conds)) == 22
    parsed = pd.DataFrame([L.parse_variant(G.variant_name("voxconverse_abcde_0001", c) + ".wav") for c in conds])
    assert parsed.axis.value_counts().to_dict() == {"duration": 8, "snr": 6, "reverb": 4, "filter": 4}
    assert set(parsed.loc[parsed.axis == "reverb", "t60_s"]) == {0.0, 0.3, 1.0, 2.5}
    assert set(parsed.loc[parsed.axis == "snr", "snr_db"]) == set(G.SNRS)
    fac = [L.parse_variant(G.variant_name("esc50_1-1000-A-3_0002", c) + ".wav") for c in G.factorial_conditions()]
    assert len(fac) == 48 and {f["axis"] for f in fac} == {"factorial"}
    assert L.recording_id("esc50_1-1000-A-3_0002_1000ms_dur20ms.wav") == "esc:1-1000"
    assert L.base_id("voxconverse_abcde_0001_1000ms_snr+0dB.wav") == "voxconverse_abcde_0001"


def test_signal_operations():
    rng = np.random.default_rng(1)
    base = G.normalize_base(rng.standard_normal(16000) + 0.3)
    assert abs(base.mean()) < 1e-6 and np.isclose(np.abs(base).max(), G.PEAK)
    seg = G.fade(G.center_cut(base, 20))
    assert len(seg) == 320 and seg[0] == pytest.approx(0, abs=0.1) and abs(seg[0]) < abs(base[16000 // 2 - 160])
    x = G.place(seg, rng)
    assert len(x) == 32000 and np.allclose(x[16000 - 160:16000 + 160], seg, atol=1e-3)
    noisy, sigma = G.add_noise(x, G.rms(seg), -10, rng)
    assert sigma == pytest.approx(G.rms(seg) * 10 ** 0.5)
    wet = G.reverberate(G.center_cut(base, 1000), tiny_rooms()[2.5][0]["rir"])
    assert G.rms(wet) == pytest.approx(G.rms(G.center_cut(base, 1000)))
    clipped, gain = G.guard(np.array([0.5, -2.0, 1.0]))
    assert gain == pytest.approx(G.CLIP_GUARD / 2.0) and np.abs(clipped).max() <= G.CLIP_GUARD + 1e-12
    a, pa = G.render(base, "snr-10dB", "voxconverse_abcde_0001")
    b, _ = G.render(base, "snr-10dB", "voxconverse_abcde_0001")
    c, _ = G.render(base, "snr-10dB", "voxconverse_abcde_0002")
    assert a.dtype == np.float32 and len(a) == 32000 and np.array_equal(a, b) and not np.array_equal(a, c)
    assert pa["noise_sigma"] == pytest.approx(pa["segment_rms"] / 10 ** (-10 / 20))
    r, pr = G.render(base, "reverb2.5s", "voxconverse_abcde_0001", tiny_rooms())
    assert pr["t60_target_s"] == 2.5 and pr["room"].startswith("t60_2.5")
    f, pf = G.render(base, "dur20ms_snr+20dB", "esc50_1-1000-A-3_0001")
    assert pf["duration_ms"] == 20 and pf["snr_db"] == 20


def test_select_build_pack_and_validate(tmp_path):
    vox, esc, esc_paths = synth_sources(tmp_path / "src")
    clips = G.select_clips(vox, esc, seed=42, counts=COUNTS, log=lambda m: None)
    again = G.select_clips(vox, esc, seed=42, counts=COUNTS, log=lambda m: None)
    pd.testing.assert_frame_equal(clips, again)
    assert clips.groupby(["split", "label"]).size().to_dict() == {
        ("dev", "NONSPEECH"): 2, ("dev", "SPEECH"): 2, ("test", "NONSPEECH"): 6, ("test", "SPEECH"): 6,
        ("train", "NONSPEECH"): 2, ("train", "SPEECH"): 2}
    for a, b in (("test", "train"), ("test", "dev"), ("train", "dev")):
        assert not set(clips[clips.split == a].recording) & set(clips[clips.split == b].recording)
    assert clips.loc[clips.split == "test"].groupby("recording").size().max() == 2
    out = tmp_path / "out"
    G.build_bank("test_bank_v2", clips[clips.split == "test"], G.test_conditions(), vox, esc_paths, tiny_rooms(),
                 out, log=None)
    man = L.load_manifest(out / "test_bank_v2" / "manifest.csv", audio_root=out / "test_bank_v2" / "audio")
    rep = banks.validate(man, {"rows": 12 * 22, "conditions": 22, "base_ids": 12, "rec_ids": 3 + 6,
                               "speech_share": 0.5}, check_audio=True)
    assert rep["ok"], rep["failures"]
    meta = pd.read_csv(out / "test_bank_v2" / "metadata.csv")
    assert {"start_s", "source_file", "segment_rms", "seed", "gain", "t60_measured_s"} <= set(meta.columns)
    assert (meta.loc[meta.condition.str.startswith("snr"), "noise_sigma"] > 0).all()
    tar, sha = G.pack("test_bank_v2", out)
    assert (out / "test_bank_v2.tar.sha256").read_text().split()[0] == sha
    banks._safe_extract(tar, tmp_path / "unpacked")
    man2 = L.load_manifest(tmp_path / "unpacked" / "test_bank_v2" / "manifest.csv",
                           audio_root=tmp_path / "unpacked" / "test_bank_v2" / "audio")
    assert set(man2.variant_id) == set(man.variant_id)


def test_rooms_with_pyroomacoustics(tmp_path):
    pytest.importorskip("pyroomacoustics")
    rooms = G.make_rooms(seed=42, per_t60=1, t60s=(0.3,), log=lambda m: None)
    r = rooms[0.3][0]
    assert abs(r["rt60"] - 0.3) <= G.RT60_TOL * 0.3 and np.isclose(np.sum(r["rir"] ** 2), 1.0)
    path = G.save_rooms(rooms, tmp_path / "rooms.npz")
    back = G.load_rooms(path)
    assert np.allclose(back[0.3][0]["rir"], r["rir"], atol=1e-6) and back[0.3][0]["rt60"] == r["rt60"]


def test_runner_builds_banks_and_the_pilot_accepts_them(tmp_path, monkeypatch):
    from erclm.gitlog import RunsLog
    from erclm.host import Host
    from erclm.runner import DEFAULTS, Runner, _merge
    from erclm.storage import LocalRemote
    vox, esc, esc_paths = synth_sources(tmp_path / "src")
    monkeypatch.setattr(G, "fetch_sources", lambda *a, **k: (vox, esc, esc_paths))
    monkeypatch.setattr(G, "make_rooms", lambda *a, **k: tiny_rooms())
    remote = LocalRemote(tmp_path / "drive")
    gh = helpers.make_github(tmp_path)
    cfg = _merge(DEFAULTS, {
        "run_name": "banks-test", "model": {"revision": "x"}, "stages": ["build_banks"],
        "banks": {"seed": 42, "names": {"test": "test_bank_v2", "train": "train_bank_v2", "dev": "dev_bank_v2"},
                  "counts": COUNTS, "sources": {"pinned": "in the test"},
                  "expect": {"test": {"rows": 264, "conditions": 22, "base_ids": 12, "rec_ids": 9, "speech_share": 0.5},
                             "train": {"rows": 192, "conditions": 48, "base_ids": 4, "speech_share": 0.5},
                             "dev": {"rows": 192, "conditions": 48, "base_ids": 4, "speech_share": 0.5}}}})
    runs = RunsLog(str(gh), None, tmp_path / "vm1" / "runs_clone", "banks-test", backoff_s=0)
    Runner(cfg, Host("local", tmp_path / "vm1"), remote, runs, session_id="b1").run()
    for name in ("test_bank_v2", "train_bank_v2", "dev_bank_v2"):
        assert remote.exists(f"data/{name}.tar") and remote.exists(f"data/{name}.tar.sha256")
        assert remote.get_json(f"data/manifests/{name}.protocol.json")["report"]["ok"]
    assert remote.exists("data/rirs/rooms_v2.npz")
    rep = json.loads(helpers.show(gh, "banks-test/banks/test_bank_v2.json"))
    assert rep["rows"] == 264 and rep["rec_ids"] == 9
    # a second session finds the banks done
    runs2 = RunsLog(str(gh), None, tmp_path / "vm2" / "runs_clone", "banks-test", backoff_s=0)
    snap = Runner(cfg, Host("local", tmp_path / "vm2"), remote, runs2, session_id="b2").run()
    assert snap["stages"]["build_banks"].get("skipped")
    # the pilot's data stage accepts the generated test bank
    pcfg = _merge(DEFAULTS, {"run_name": "pilot-test", "model": {"revision": "x"},
                             "data": {"test": {"tar": "data/test_bank_v2.tar", "manifest": "test_bank_v2/manifest.csv",
                                               "audio": "test_bank_v2/audio",
                                               "expect": {"rows": 264, "conditions": 22, "speech_share": 0.5}}}})
    runs3 = RunsLog(str(gh), None, tmp_path / "vm3" / "runs_clone", "pilot-test", backoff_s=0)
    snap = Runner(pcfg, Host("local", tmp_path / "vm3"), remote, runs3, session_id="p1", only=["data_test"]).run()
    assert snap["stages"]["data_test"]["status"] == "done"
