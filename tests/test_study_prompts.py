"""The registered reading of the study prompts on Qwen2-Audio (tools/study_prompts.py), on fake stores.

Run: python -m pytest -q tests/test_study_prompts.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import study_prompts as pe  # noqa: E402
from test_omni_control import _items, _store  # noqa: E402


def build(root: Path):
    rng = np.random.default_rng(0)
    m = _items()
    y = m["label"].to_numpy()
    base = rng.standard_normal(len(m)) + 2.0 * y - 4.0     # well ordered, threshold far off
    lora = rng.standard_normal(len(m)) + 2.5 * y - 1.0
    _store(root / "q2_base", m, base)
    _store(root / "q2_lora", m, lora)
    _store(root / "q2_base_ba", m, base + 0.3 * rng.standard_normal(len(m)) + 3.0)
    _store(root / "q2_lora_ba", m, lora + 0.3 * rng.standard_normal(len(m)))
    _store(root / "opro_llm_base_base", m, base + 4.0)        # same order, moved threshold: criterion
    _store(root / "hand_csl_base", m, base)                   # no change
    _store(root / "opro_tmpl_base", m, base + 0.5)
    _store(root / "opro_llm_base_lora", m, lora)
    _store(root / "hand_csl_lora", m, lora)
    _store(root / "opro_tmpl_lora", m, rng.standard_normal(len(m)) + 5.0 * y - 2.5)  # better ordering


def test_reading_on_fake_stores(tmp_path):
    build(tmp_path)
    res = pe.main(tmp_path, B=60, seed=1)
    d = res["base/opro_llm_base/duration"]
    assert abs(d["O"]["value"]) < 1e-9 and d["G"]["value"] > 0.2
    assert d["reading"] == "mostly criterion" and res["primary"]["base"]["duration"] == "mostly criterion"
    assert res["base/hand_csl/duration"]["reading"] == "the prompt does not improve the native BA"
    lo = res["lora/opro_tmpl/duration"]
    assert lo["dAUROC"]["value"] > 0 and lo["O"]["value"] > 0
    assert res["base/opro_llm_base/short"]["reading"] is None
    assert "cross_model" not in res
    assert (tmp_path / "reading_prompts" / "prompts.csv").exists()


def test_reading_rule():
    v = lambda x, lo, hi: {"value": x, "ci": [lo, hi]}  # noqa: E731
    assert pe.reading({"G": v(0.1, 0.01, 0.2), "share_C": v(0.9, 0.7, 1.0)}) == "the prompt does not improve the native BA"
    assert pe.reading({"G": v(0.2, 0.1, 0.3), "share_C": v(0.9, 0.7, 1.0)}) == "mostly criterion"
    assert pe.reading({"G": v(0.2, 0.1, 0.3), "share_C": v(0.2, 0.0, 0.4)}) == "mostly ranking"
    assert pe.reading({"G": v(0.2, 0.1, 0.3), "share_C": v(0.5, 0.3, 0.7)}) == "mixed"
