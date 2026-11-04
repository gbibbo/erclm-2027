"""The registered reading of the excerpt-position sweep in Qwen3-Omni (tools/omni_seam.py), on fake stores.

Run: python -m pytest -q tests/test_omni_seam.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import omni_seam as os_  # noqa: E402
from test_omni_control import _items, _store  # noqa: E402


def build(root: Path, seam_hurts: bool = True):
    rng = np.random.default_rng(0)
    m = _items()
    y = m["label"].to_numpy()
    sep = {sh: (0.3 if (sh == 0 and seam_hurts) else 1.5) for sh in os_.SHIFTS}
    for sh, (where, track) in os_.SHIFTS.items():
        _store(root / where / track, m, rng.standard_normal(len(m)) + sep[sh] * y - 2.0, ("enc_region_mean",), sep=1.0)
    _store(root / "omni" / "hand_csl", m, rng.standard_normal(len(m)) + 0.5 * y)
    _store(root / "seam" / "hand_csl_shift500", m, rng.standard_normal(len(m)) + 1.8 * y)
    _store(root / "q2_base", m, rng.standard_normal(len(m)) + 2.5 * y - 3.0)


def test_reading_when_the_seam_hurts(tmp_path):
    build(tmp_path)
    res = os_.main(tmp_path, B=40, seed=1)
    r = res["R5"]
    assert r["R5a"] == "the seam hurts at every position tested"
    assert r["R5b"] == "the position within the block matters, not the position in the container"
    assert r["R5c"]["splitting the crop across blocks explains most of it"]
    assert res["hand"]["hand_500 - ab_500"]["value"] > 0
    assert (tmp_path / "reading_seam" / "seam.csv").exists()


def test_reading_when_it_does_not(tmp_path):
    build(tmp_path, seam_hurts=False)
    r = os_.main(tmp_path, B=40, seed=1, with_probes=False)["R5"]
    assert r["R5a"].startswith("the rotation improves only at")
    assert not r["R5c"]["splitting the crop across blocks explains most of it"]
