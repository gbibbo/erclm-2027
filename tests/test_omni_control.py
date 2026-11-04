"""The registered reading of Qwen3-Omni on the 300 control clips (tools/omni_control.py), on fake stores.

Run: python -m pytest -q tests/test_omni_control.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import omni_control as oc  # noqa: E402
from erclm import lib_v2 as L  # noqa: E402

N_PER_CLASS = 10


def _items():
    rows = []
    for i in range(2 * N_PER_CLASS):
        lab = int(i < N_PER_CLASS)
        for c in oc.DUR + oc.SNR:
            rows.append({"variant_id": f"b{i}_{c}", "base_id": f"b{i}", "rec_id": f"r{i}", "label": lab,
                         "condition": c})
    return pd.DataFrame(rows)


def _store(d: Path, meta: pd.DataFrame, s: np.ndarray, features=(), dim=6, seed=0, sep=0.0):
    rng = np.random.default_rng(seed)
    st = L.FeatureStore(d, shard_size=97)
    y = meta["label"].to_numpy()
    for i, row in enumerate(meta.to_dict("records")):
        arr = {f: (rng.standard_normal(dim) + sep * y[i]).astype(np.float16) for f in features}
        st.add({**row, "s_primary": float(s[i]), "top1_is_label": True}, arr)
    st.flush()


def _score(meta, rng, short_sep, long_sep, offset=0.0):
    y = meta["label"].to_numpy()
    short = meta["condition"].isin(oc.SHORT).to_numpy()
    return rng.standard_normal(len(meta)) + np.where(short, short_sep, long_sep) * y + offset


def build(root: Path):
    rng = np.random.default_rng(0)
    m = _items()
    dur = m[m["condition"].isin(oc.DUR)].reset_index(drop=True)
    _store(root / "omni" / "ab", m, _score(m, rng, 0.2, 3.0, -2.0), oc.Q3_PROBES, sep=0.5)
    _store(root / "omni" / "ba_speech_second", m, _score(m, rng, 0.2, 3.0, -1.0))
    _store(root / "omni" / "hand_csl", m, _score(m, rng, 0.5, 3.0, -1.0))
    _store(root / "omni" / "ab_shift500", dur, _score(dur, rng, 2.0, 3.0, -2.0), oc.Q3_PROBES, sep=1.5)
    extra = m.assign(variant_id=m["variant_id"] + "_reverb", condition="reverb_1.0s").iloc[:20]
    q2 = pd.concat([m, extra], ignore_index=True)   # the full-v1 stores hold more items than the 300 clips
    _store(root / "q2_base", q2, _score(q2, rng, 2.0, 3.0, -3.0), oc.Q2_PROBES, sep=1.5)
    _store(root / "q2_lora", q2, _score(q2, rng, 2.5, 4.0, 0.0))
    _store(root / "q2_base_ba", m, _score(m, rng, 2.0, 3.0, 1.0))
    _store(root / "q2_lora_ba", m, _score(m, rng, 2.0, 3.0, 0.0))


def test_reading_on_fake_stores(tmp_path):
    build(tmp_path)
    res = oc.main(tmp_path, B=40, seed=1, expect=2 * N_PER_CLASS * 14)
    assert set(res["R1"]) == {"ab/dur_20ms", "ab/dur_40ms", "s_sim/dur_20ms", "s_sim/dur_40ms"}
    assert res["R2"]["q3_ab - q2_base, 20 and 40 ms"]["value"] < 0
    assert res["R3"]["delta"]["value"] > 0 and res["R3"]["gap"] > 0
    assert res["R4"]["a"]["value"] < 0
    lo, hi = res["R2"]["q3_ab - q2_base, 20 and 40 ms"]["ci"]
    assert lo <= res["R2"]["q3_ab - q2_base, 20 and 40 ms"]["value"] <= hi
    saved = json.loads((tmp_path / "reading" / "reading.json").read_text(encoding="utf-8"))
    assert saved["R3"]["reading"] == res["R3"]["reading"]
    assert {"duration", "snr"} == set(saved["descriptive"]["recipe"])
    t = pd.read_csv(tmp_path / "reading" / "paired_all.csv")
    assert {"short", "duration", "snr", "dur_20ms", "snr_-10dB"} <= set(t["group"])


def test_folds_are_shared_between_models(tmp_path):
    """Two models on the same aligned items get the same folds, so their probes are paired."""
    from sklearn.model_selection import StratifiedGroupKFold
    m = _items()
    y, g = m["label"].to_numpy(int), m["rec_id"].to_numpy()
    rng = np.random.default_rng(0)
    folds = [[tuple(te) for _, te in StratifiedGroupKFold(5, shuffle=True, random_state=0)
              .split(rng.standard_normal((len(m), d)), y, g)] for d in (3, 9)]
    assert folds[0] == folds[1]


def test_rule_readings():
    d = lambda v, lo, hi: {"value": v, "ci": [lo, hi]}  # noqa: E731
    assert oc.r3_reading(d(0.10, 0.04, 0.16), 0.17) == "the seam explains most of it"
    assert oc.r3_reading(d(0.03, 0.01, 0.05), 0.17) == "the seam explains a minor part"
    assert oc.r3_reading(d(0.01, -0.02, 0.04), 0.17) == "the seam does not explain it"
    assert oc.r3_reading(d(0.05, -0.02, 0.12), 0.17) == "no conclusion"
    assert oc.r3_reading(d(0.05, 0.01, 0.09), -0.01) == "no gap, only delta is reported"
    assert oc.r4_reading(d(-0.10, -0.15, -0.05), d(0.0, -0.1, 0.1)).startswith("representation")
    assert oc.r4_reading(d(0.00, -0.01, 0.02), d(0.10, 0.06, 0.14)).startswith("readout")
    assert oc.r4_reading(d(0.00, -0.03, 0.02), d(0.10, 0.06, 0.14)) == "no conclusion"
