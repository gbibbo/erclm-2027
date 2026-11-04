"""The registered generation readout of the study prompts (tools/study_prompts_generation.py), on fake stores.

Run: python -m pytest -q tests/test_study_prompts_generation.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import study_prompts_generation as pg  # noqa: E402
from erclm import lib_v2 as L  # noqa: E402
from test_omni_control import _items, _store  # noqa: E402

SP, NS, THIS = 50, 45, 1986   # "S", "N" and "This"


def _gen_store(d: Path, m, rng, p_correct: float, starts_this: float = 0.0, s_bias: float = 0.0):
    """Answers right with probability p_correct, some starting with "This", and a first-token score
    biased towards "S" by s_bias, as the base model does with the OPRO-LLM prompt."""
    st = L.FeatureStore(d, shard_size=97)
    y = m["label"].to_numpy()
    for i, row in enumerate(m.to_dict("records")):
        speech = bool(y[i]) if rng.random() < p_correct else not y[i]
        this = rng.random() < starts_this
        first = THIS if this else (SP if speech else NS)
        answer = ("This is speech." if speech else "This is non-speech.") if this else ("SPEECH" if speech else "NONSPEECH")
        ids = np.r_[first, [t for t in (SP, NS, THIS, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
                            if t != first]][:20].astype(np.int32)
        logits = np.sort(rng.standard_normal(20))[::-1].astype(np.float32) + 10
        s = (1.0 if speech else -1.0) + s_bias + rng.normal(0, 0.5)
        st.add({**row, "s_primary": s, "answer": answer, "gen_label": "SPEECH" if speech else "NONSPEECH",
                "gen_speech": speech, "n_new_tokens": 3}, {"first_top_ids": ids, "first_top_logits": logits})
    st.flush()


def build(root: Path):
    rng = np.random.default_rng(0)
    m = _items()
    y = m["label"].to_numpy()
    _store(root / "q2_base", m, rng.standard_normal(len(m)) + 2.0 * y - 4.0)
    _store(root / "q2_lora", m, rng.standard_normal(len(m)) + 2.5 * y - 1.0)
    _store(root / "q2_base_ba", m, rng.standard_normal(len(m)) + 2.0 * y)
    _store(root / "q2_lora_ba", m, rng.standard_normal(len(m)) + 2.5 * y)
    for p in pg.PROMPTS:
        _gen_store(root / f"gen_{p}_base", m, rng, 0.85, starts_this=0.5 if p == "opro_llm_base" else 0.0,
                   s_bias=2.5 if p == "opro_llm_base" else 0.0)
        _gen_store(root / f"gen_{p}_lora", m, rng, 0.9)


def test_reading_on_fake_stores(tmp_path):
    build(tmp_path)
    res = pg.main(tmp_path, B=40, seed=1)
    assert res["primary"]["registered"]["base/opro_llm_base/duration"]["first_token_misreads"]
    assert not res["primary"]["registered"]["lora/opro_tmpl/duration"]["first_token_misreads"]
    d = res["registered/base/opro_llm_base/duration"]
    assert 0.7 < d["BA_text"]["value"] < 0.95 and d["BA_L0_first"]["value"] < d["BA_text"]["value"]
    assert res["token_sets"]["registered/base/opro_llm_base"] == {"speech": [SP], "nonspeech": [NS]}
    assert res["S_token"]["registered/lora/opro_tmpl"]["share_parsed_nonspeech"] == 0.0
    assert res["primary"]["registered"]["lora_S_is_not_speech"] is False
    assert res["parser_agreement"]["base/opro_llm_base"] == 1.0  # the fake answers parse alike
    assert res["primary"]["qwen2_parser"] == res["primary"]["registered"]
    assert (tmp_path / "reading_generated" / "generated.csv").exists()


def test_gen_score_counts_missing_tokens_below_the_top():
    ids = np.array([[50, 45, 7], [7, 8, 9]])
    logits = np.array([[3.0, 1.0, 0.0], [5.0, 4.0, 2.0]])
    s = pg.gen_score(ids, logits, [50], [45])
    assert s[0] == 2.0 and s[1] == 0.0


def test_partial_reading_matches_the_full_one(tmp_path):
    import shutil
    build(tmp_path)
    full = pg.main(tmp_path, B=40, seed=1, out=tmp_path / "full")
    for d in ("gen_opro_llm_base_lora", "gen_hand_csl_base", "gen_hand_csl_lora"):
        shutil.rmtree(tmp_path / d)
    part = pg.main(tmp_path, B=40, seed=1, out=tmp_path / "part", partial=True)
    assert part["missing"] == ["base/hand_csl", "lora/opro_llm_base", "lora/hand_csl"]
    for k in ("registered/base/opro_llm_base/duration", "qwen2_parser/lora/opro_tmpl/snr"):
        assert part[k] == full[k]
    assert "registered/lora/hand_csl/duration" not in part
    assert set(part["primary"]["registered"]) == set(full["primary"]["registered"])
