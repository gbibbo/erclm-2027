"""The repetition penalty on the first generated letter (tools/repetition_penalty.py), on fake stores.

Run: python -m pytest -q tests/test_repetition_penalty.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import repetition_penalty as pz  # noqa: E402
from erclm import lib_v2 as L  # noqa: E402
from test_omni_control import _items, _store  # noqa: E402


def _gen_store(d: Path, m, rng, s_offset: float):
    """The letters "S" and "N" at the top with logits near 10, "S" raised by s_offset, and the written letter
    taken from the penalised logits, as the decoding of the study under review does."""
    st = L.FeatureStore(d, shard_size=97)
    y = m["label"].to_numpy()
    for i, row in enumerate(m.to_dict("records")):
        zs = 10.0 + (1.0 if y[i] else -1.0) + s_offset + rng.normal(0, 0.8)
        zn = 10.0
        ids = np.r_[pz.S, pz.N, np.arange(100, 118)].astype(np.int32)
        logits = np.r_[zs, zn, np.linspace(5, 1, 18)].astype(np.float32)
        order = np.argsort(-logits)
        speech = zs / pz.PENALTY > zn
        st.add({**row, "s_primary": zs - zn, "answer": "SPEECH" if speech else "NONSPEECH",
                "gen_label": "SPEECH" if speech else "NONSPEECH"},
               {"first_top_ids": ids[order], "first_top_logits": logits[order]})
    st.flush()


def test_penalize_matches_hf():
    ids = np.array([[pz.S, pz.N, 7]])
    out = pz.penalize(ids, np.array([[2.2, 2.0, -1.0]]))
    assert np.allclose(out, [[2.0, 2.0, -1.0]])
    assert np.allclose(pz.penalize(np.array([[pz.S]]), np.array([[-1.0]])), [[-1.1]])


def test_reading_on_fake_stores(tmp_path):
    rng = np.random.default_rng(0)
    m = _items()
    _store(tmp_path / "q2_base_ba", m, rng.standard_normal(len(m)))
    for p in pz.PROMPTS:
        _gen_store(tmp_path / f"gen_{p}_base", m, rng, s_offset=1.5)   # liberal raw, the penalty helps
        _gen_store(tmp_path / f"gen_{p}_lora", m, rng, s_offset=0.5)
    res = pz.main(tmp_path, B=30, seed=1)
    for p in pz.PROMPTS:
        for b in ("base", "lora"):
            assert res[f"{b}/{p}"]["penalized_predicts_written"] == 1.0
            assert res[f"{b}/{p}"]["raw_S_written_N"] > 0
    assert res["duration/base_hand_csl_written - base_hand_csl_raw"]["value"] > 0
    assert abs(res["duration/base_opro_tmpl_raw - base_hand_csl_raw"]["value"]) < 0.1
