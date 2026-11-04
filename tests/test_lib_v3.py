"""Tests for lib_v3, the exploratory analyses for closing the paper."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from erclm import lib_v3 as L  # noqa: E402


def test_merge_intervals_joins_overlaps_and_drops_empty():
    assert L.merge_intervals([(2, 3), (0, 1), (0.5, 1.5), (4, 4)]) == [(0, 1.5), (2, 3)]


def test_interval_coverage():
    iv = [(0.0, 1.0), (2.0, 3.0)]
    assert L.interval_coverage(iv, 0.5, 1.5) == pytest.approx(0.5)
    assert L.interval_coverage(iv, 0.0, 3.0) == pytest.approx(2 / 3)
    assert L.interval_coverage(iv, 1.2, 1.8) == 0.0


def test_crop_bounds_matches_center_cut():
    a, b = L.crop_bounds(10.0, 20)
    assert a == pytest.approx(10.0 + (16000 - 320) // 2 / 16000)
    assert b - a == pytest.approx(0.02)
    a, b = L.crop_bounds(10.0, 1000)
    assert (a, b) == pytest.approx((10.0, 11.0))


def _meta(rows):
    return pd.DataFrame(rows, columns=["audio_path", "base_id", "dataset", "recording", "condition",
                                       "start_s", "duration_ms", "ground_truth", "segment_rms"])


def test_speech_coverage_reads_rttm(tmp_path):
    # Speech from 0 to 0.5 s and from 0.6 to 2 s. A 1000 ms clip starting at 0 has its
    # 100 ms centre crop at 0.45 to 0.55 s, half of it speech.
    (tmp_path / "rec.rttm").write_text(
        "SPEAKER rec 1 0.0 0.5 <NA> <NA> spk1 <NA> <NA>\n"
        "SPEAKER rec 1 0.6 1.4 <NA> <NA> spk2 <NA> <NA>\n")
    meta = _meta([
        ["voxconverse_rec_0000_1000ms_dur100ms.wav", "b0", "voxconverse", "rec", "dur100ms", 0.0, 100, "SPEECH", 0.1],
        ["voxconverse_rec_0000_1000ms_dur1000ms.wav", "b0", "voxconverse", "rec", "dur1000ms", 0.0, 1000, "SPEECH", 0.1],
        ["esc50_x_0000_1000ms_dur100ms.wav", "b1", "esc50", "x", "dur100ms", 0.0, 100, "NONSPEECH", 0.1],
    ])
    cov = L.speech_coverage(meta, [tmp_path]).set_index("duration_ms")["coverage"]
    assert cov[100] == pytest.approx(0.5)
    assert cov[1000] == pytest.approx(0.9)
    tab = L.coverage_table(L.speech_coverage(meta, [tmp_path]))
    assert tab.loc[100, "at_least_half"] == 1.0 and tab.loc[100, "full"] == 0.0


def test_speech_coverage_missing_rttm_fails(tmp_path):
    meta = _meta([["voxconverse_r_0000_1000ms_dur20ms.wav", "b0", "voxconverse", "r", "dur20ms", 0.0, 20, "SPEECH", 0.1]])
    with pytest.raises(FileNotFoundError):
        L.speech_coverage(meta, [tmp_path])


def test_relative_level_db():
    meta = _meta([
        ["a_dur20ms.wav", "a", "voxconverse", "r", "dur20ms", 0.0, 20, "SPEECH", 0.01],
        ["a_dur1000ms.wav", "a", "voxconverse", "r", "dur1000ms", 0.0, 1000, "SPEECH", 0.1],
        ["a_snr+0dB.wav", "a", "voxconverse", "r", "snr+0dB", 0.0, 1000, "SPEECH", 0.1],
    ])
    rel = L.relative_level_db(meta).set_index("variant_id")["rel_db"]
    assert list(rel.index) == ["a_dur20ms", "a_dur1000ms"]
    assert rel["a_dur20ms"] == pytest.approx(-20.0)
    tab = L.level_table(L.relative_level_db(meta))
    assert tab.loc[tab["duration_ms"] == 20, "share_below"].item() == 0.0


def test_level_auroc_by_condition_is_direction_free():
    rms = [0.1, 0.2, 0.3, 0.4]
    meta = pd.DataFrame({"condition": ["c"] * 4, "segment_rms": rms,
                         "ground_truth": ["NONSPEECH", "NONSPEECH", "SPEECH", "SPEECH"]})
    out = L.level_auroc_by_condition(meta)
    assert out.loc["c", "auroc"] == 1.0 and out.loc["c", "speech_louder"]
    meta["ground_truth"] = meta["ground_truth"][::-1].to_numpy()
    out = L.level_auroc_by_condition(meta)
    assert out.loc["c", "auroc"] == 1.0 and not out.loc["c", "speech_louder"]


def test_balanced_accuracy_and_by_condition():
    assert L.balanced_accuracy([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.75)
    assert np.isnan(L.balanced_accuracy([1, 1], [1, 0]))
    sc = pd.DataFrame({"condition": ["c"] * 4, "label": [1, 1, 0, 0],
                       "R__score": [2.0, -1.0, -2.0, -3.0], "R__thr": [0.0] * 4})
    assert L.ba_by_condition(sc, ["R"]).loc["c", "R"] == pytest.approx(0.75)
    keep = pd.Series([True, False, True, True])
    assert L.ba_by_condition(sc, ["R"], keep).loc["c", "R"] == pytest.approx(1.0)


def test_ab_prompt_reproduces_the_confirmatory_prompt():
    from erclm import lib_v2
    assert L.ab_prompt("A", True)["user"] == lib_v2.PROMPTS["ab"]["user"]
    assert L.ab_prompt("A", True)["system"] == lib_v2.PROMPTS["ab"]["system"]
    assert tuple(L.ab_prompt("A", True)["labels"]) == tuple(lib_v2.PROMPTS["ab"]["labels"])


def test_prompt_variants_cross_letter_and_position():
    p = L.PROMPTS["ba_speech_first"]
    assert p["labels"] == ("B", "A") and p["user"].index("B) SPEECH") < p["user"].index("A) NONSPEECH")
    p = L.PROMPTS["ab_speech_second"]
    assert p["labels"] == ("A", "B") and p["user"].index("B) NONSPEECH") < p["user"].index("A) SPEECH")
    p = L.PROMPTS["ba_speech_second"]
    assert p["labels"] == ("B", "A") and p["user"].index("A) NONSPEECH") < p["user"].index("B) SPEECH")


def test_register_prompts_adds_without_overwriting():
    table = {"ab": {"x": 1}}
    L.register_prompts(table)
    assert table["ab"] == {"x": 1} and set(L.PROMPTS) <= set(table)
    with pytest.raises(ValueError):
        L.register_prompts({"ba_speech_first": {"other": True}})


def test_fit_threshold_maximises_balanced_accuracy():
    assert L.fit_threshold([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1]) == pytest.approx(1.5)
    # unbalanced classes: BA, not accuracy, decides
    s = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    y = [0, 0, 0, 0, 0, 1, 1]
    assert 0.4 < L.fit_threshold(s, y) < 0.5


def test_axis_mean_contrasts_average_over_conditions():
    sc = pd.DataFrame({"axis": ["a", "a", "b"], "condition": ["c1", "c2", "c3"]})
    out = L.axis_mean_contrasts(sc, {"d": {"R1": 1, "oracle:R2": -1}})
    assert set(out) == {"a:d", "b:d"}
    assert sorted(out["a:d"]) == sorted([(0.5, "R1", "c1", "ba"), (0.5, "R1", "c2", "ba"),
                                         (-0.5, "R2", "c1", "oracle_ba"), (-0.5, "R2", "c2", "oracle_ba")])


def _head_data(n=240, d=6, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    h = rng.standard_normal((n, d)).astype(np.float32)
    h[:, 0] += 1.5 * y                      # evidence the native score misses
    s = 0.8 * y + rng.standard_normal(n)    # a weak native score
    groups = np.arange(n) // 4
    return h, s, y, groups


def test_residual_and_anchored_heads_improve_on_a_weak_native_score():
    h, s, y, groups = _head_data()
    base = L.auroc(y, s)
    res = L.fit_residual_head(h, s, y, groups, Cs=(0.01, 1.0))
    anc = L.fit_anchored_head(h, s, y, groups, Cs=(0.01, 1.0))
    assert res["alpha"] == 1.0 and res["delta"].shape == (6,)
    assert L.auroc(y, L.anchored_scores(res, h, s)) > base
    assert L.auroc(y, L.anchored_scores(anc, h, s)) > base


def test_residual_head_with_strong_penalty_keeps_the_native_ranking():
    h, s, y, groups = _head_data()
    res = L.fit_residual_head(h, s, y, groups, Cs=(1e-8,))
    assert np.abs(res["delta"]).max() < 1e-4
    assert L.auroc(y, L.anchored_scores(res, h, s)) == pytest.approx(L.auroc(y, s), abs=1e-6)


def test_esc_class_group_and_cluster_rate():
    assert L.esc_class("esc50_1-100032-A-0_0007") == 0
    assert L.esc_class("esc50_5-200334-A-48_0123") == 48
    assert L.esc_group(21) == "human vocal" and L.esc_group(22) == "human other" and L.esc_group(0) == "other"
    assert L.cluster_rate_ci([1, 1, 1, 1], ["a", "a", "b", "b"], B=50) == (1.0, 1.0, 1.0)
    est, lo, hi = L.cluster_rate_ci([1, 0, 1, 0], ["a", "a", "b", "b"], B=50)
    assert est == 0.5 and lo == hi == 0.5


def test_load_store_features_reads_selected_keys_and_dedupes(tmp_path):
    for i, ids in enumerate((["a", "b"], ["b", "cf"])):
        meta = pd.DataFrame({"variant_id": ids, "role": ["test" if v != "cf" else "content_free" for v in ids],
                             "label": [1, 0]})
        meta.to_csv(tmp_path / f"shard_{i:05d}.csv", index=False)
        np.savez(tmp_path / f"shard_{i:05d}.npz", h=np.full((2, 3), i, dtype=np.float16),
                 other=np.zeros((2, 5), dtype=np.float16))
    meta, arrays = L.load_store_features(tmp_path, ["h"])
    assert list(meta["variant_id"]) == ["a", "b"] and set(arrays) == {"h"}
    assert arrays["h"].dtype == np.float32 and arrays["h"][1, 0] == 1.0     # last copy of 'b'
    meta, _ = L.load_store_features(tmp_path, drop_content_free=False)
    assert list(meta["variant_id"]) == ["a", "b", "cf"]


# ---------------------------------------------------------------- GPU interventions (22 Oct)

def test_bandpass_keeps_the_band_and_removes_the_edges():
    sr = 16000
    t = np.arange(2 * sr) / sr
    y = L.bandpass(np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 7500 * t) + np.sin(2 * np.pi * 20 * t))
    assert y.dtype == np.float32 and len(y) == len(t)
    mid = slice(sr // 2, 3 * sr // 2)

    def amp(f):
        return 2 * np.abs(np.mean(y[mid] * np.exp(-2j * np.pi * f * t[mid])))
    assert amp(1000) == pytest.approx(1.0, abs=0.01)
    assert amp(7500) < 1e-3 and amp(20) < 1e-3


def test_transform_audio_writes_each_file_once(tmp_path):
    import soundfile as sf
    rng = np.random.default_rng(0)
    rows = []
    for i in range(3):
        p = tmp_path / f"in{i}.wav"
        sf.write(str(p), (rng.standard_normal(32000) * 0.1).astype(np.float32), 16000)
        rows.append({"variant_id": f"v{i}", "path": str(p), "label": i % 2})
    man = pd.DataFrame(rows)
    spec = {"kind": "bandpass", "lo": 60, "hi": 6000, "order": 8}
    out = L.transform_audio(man, tmp_path / "t", spec)
    assert list(out.columns) == list(man.columns) and (out["label"] == man["label"]).all()
    assert (man["path"] == [r["path"] for r in rows]).all()          # the input manifest is untouched
    x, sr = sf.read(out["path"][0], dtype="float32")
    assert sr == 16000
    np.testing.assert_allclose(x, L.bandpass(sf.read(rows[0]["path"], dtype="float32")[0]), atol=1e-6)
    mtimes = [Path(p).stat().st_mtime_ns for p in out["path"]]
    again = L.transform_audio(man, tmp_path / "t", spec)
    assert [Path(p).stat().st_mtime_ns for p in again["path"]] == mtimes
    assert not list((tmp_path / "t").glob("*.partial.wav"))


def test_parse_lora_keep():
    assert L.parse_lora_keep("audio_tower") == ("audio_tower", None)
    assert L.parse_lora_keep("language_model:16-31") == ("language_model", (16, 31))
    assert L.parse_lora_keep("language_model:5") == ("language_model", (5, 5))
    for bad in ("decoder", "language_model:3-1"):
        with pytest.raises(ValueError):
            L.parse_lora_keep(bad)


def _tiny_lora():
    from peft import LoraConfig, get_peft_model
    from test_model_v2 import PROC, _randomize_lora, make_model
    pm = get_peft_model(make_model(PROC), LoraConfig(r=4, lora_alpha=16, lora_dropout=0.0,
                                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    _randomize_lora(pm)
    return pm.eval()


def _collect(model):
    import torch
    from erclm import lib_v2 as L2
    from test_model_v2 import PROC
    wav = (np.random.default_rng(2).standard_normal(32000) * 0.05).astype(np.float32)
    info = L2.label_token_info(PROC.tokenizer, ("A", "B"))
    inp = L2.prepare_inputs(PROC, L2.build_text(PROC, L2.PROMPTS["ab"]), wav)
    rec = L2.Recorder(model)
    with torch.inference_mode():
        s, a = L2.forward_collect(model, rec, inp, info, 200, len(wav))
    rec.remove()
    return s, a


def test_restrict_lora_switches_off_the_other_parts():
    from test_model_v2 import PROC, make_model
    _, base = _collect(make_model(PROC))
    _, full = _collect(_tiny_lora())

    llm = _tiny_lora()
    rep = L.restrict_lora(llm, "language_model")
    assert rep["kept"] == {"language_model": 12} and rep["disabled"] == {"audio_tower": 6}
    _, a = _collect(llm)
    np.testing.assert_allclose(a["enc_global_mean"], base["enc_global_mean"], atol=1e-6)  # encoder as in base
    assert not np.allclose(a["h_final"], base["h_final"]) and not np.allclose(a["h_final"], full["h_final"])

    enc = _tiny_lora()
    assert L.restrict_lora(enc, "audio_tower")["kept"] == {"audio_tower": 6}
    _, a = _collect(enc)
    np.testing.assert_allclose(a["enc_global_mean"], full["enc_global_mean"], atol=1e-6)  # encoder as in LoRA
    assert not np.allclose(a["enc_global_mean"], base["enc_global_mean"])

    upper = _tiny_lora()
    rep = L.restrict_lora(upper, "language_model:1-2")
    assert rep["kept"] == {"language_model": 8} and rep["disabled"] == {"audio_tower": 6, "language_model": 4}
    with pytest.raises(ValueError):
        L.restrict_lora(_tiny_lora(), "language_model:50-60")


def test_restrict_lora_keeps_several_parts():
    """I3 (23 Oct): the encoder together with one half of the language model."""
    both = _tiny_lora()
    rep = L.restrict_lora(both, "audio_tower+language_model:1-2")
    assert rep["kept"] == {"audio_tower": 6, "language_model": 8} and rep["disabled"] == {"language_model": 4}
    _, a = _collect(both)
    _, full = _collect(_tiny_lora())
    np.testing.assert_allclose(a["enc_global_mean"], full["enc_global_mean"], atol=1e-6)  # encoder as in LoRA
    assert not np.allclose(a["h_final"], full["h_final"])                                  # layer 0 is off
    every = _tiny_lora()
    assert L.restrict_lora(every, "audio_tower+language_model")["disabled"] == {}
    _, a = _collect(every)
    np.testing.assert_allclose(a["h_final"], full["h_final"], atol=1e-5)                   # same as the full LoRA
    with pytest.raises(ValueError):
        L.restrict_lora(_tiny_lora(), "audio_tower+decoder")


def test_study_prompts_match_the_omni_hand_prompt():
    from erclm import lib_v2, omni
    assert L.PROMPTS["hand_csl"] == omni.PROMPTS["hand_csl"]
    assert L.PROMPTS["opro_llm_base"]["labels"] == ("SPEECH", "NON-SPEECH")
    assert "opro_tmpl" in lib_v2.PROMPTS and "opro_tmpl" not in L.PROMPTS
