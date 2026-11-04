"""Tests for bankgen_v3, the provenance bank of the closing plan (decision D2)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from erclm import bankgen as G  # noqa: E402
from erclm import bankgen_v3 as P  # noqa: E402

SR = G.SR


def _recording(tmp_path, name, speech, total_s=12.0, quiet=(), seed=0):
    """A WAV whose RTTM speech regions and a loud 'music' region are noise at known levels."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(total_s * SR)) * 0.05          # non-speech background, audible
    for a, b in quiet:
        x[int(a * SR):int(b * SR)] *= 1e-3                     # near silence
    for a, b in speech:
        x[int(a * SR):int(b * SR)] = rng.standard_normal(int((b - a) * SR)) * 0.1
    import soundfile as sf
    wav = tmp_path / f"{name}.wav"
    sf.write(str(wav), x, SR)
    rttm = tmp_path / f"{name}.rttm"
    rttm.write_text("".join(f"SPEAKER {name} 1 {a} {b - a} <NA> <NA> spk <NA> <NA>\n" for a, b in speech))
    return {"wav": wav, "rttm": rttm}


def test_nonspeech_candidates_keep_margin_and_skip_silence(tmp_path):
    rec = _recording(tmp_path, "r1", speech=[(0.0, 4.0)], quiet=[(8.0, 12.0)])
    audio = G.load_audio(rec["wav"])
    cands = P.nonspeech_candidates(audio, G.read_rttm(rec["rttm"]), min_rms=0.01)
    assert cands, "the loud non-speech stretch should give windows"
    assert min(cands) >= 4.0 + P.NS_MARGIN_S - 1e-9
    assert all(s < 8.0 for s in cands)                        # windows inside the quiet stretch are not taken


def test_select_provenance_clips_balances_classes_per_recording(tmp_path):
    vox = {"keep": _recording(tmp_path, "keep", speech=[(0.0, 5.0)], seed=1),
           "nons": _recording(tmp_path, "nons", speech=[(0.0, 11.5)], seed=2),
           "used": _recording(tmp_path, "used", speech=[(0.0, 5.0)], seed=3)}
    clips = P.select_provenance_clips(vox, exclude={"used"}, seed=0, per_recording=1, log=lambda *_: None)
    assert set(clips["recording"]) == {"keep"}                 # 'nons' has no non-speech window
    assert sorted(clips["label"]) == ["NONSPEECH", "SPEECH"]
    assert clips["base_id"].str.startswith("voxconverse_keep_").all()
    assert clips["base_id"].is_unique
    sp = clips.loc[clips.label == "SPEECH", "start_s"].item()
    ns = clips.loc[clips.label == "NONSPEECH", "start_s"].item()
    assert sp + 1.0 <= 5.0 + 1e-9 and ns >= 5.0 + P.NS_MARGIN_S - 1e-9


def test_provenance_bank_renders_with_bankgen(tmp_path):
    vox = {"keep": _recording(tmp_path, "keep", speech=[(0.0, 5.0)], seed=1)}
    clips = P.select_provenance_clips(vox, exclude=set(), seed=0, log=lambda *_: None)
    summary = G.build_bank("prov_test", clips, P.duration_conditions(), vox, {}, None, tmp_path / "out", log=None)
    assert summary["variants"] == 2 * 8
    man = pd.read_csv(tmp_path / "out" / "prov_test" / "manifest.csv")
    assert man["ground_truth"].value_counts().to_dict() == {"SPEECH": 8, "NONSPEECH": 8}
